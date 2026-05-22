import json
import secrets
import sys
import time
import requests
import hashlib
import base64
from uuid import uuid4
from typing import Any, Dict, TYPE_CHECKING, Optional
from urllib.parse import urlencode, urlparse
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import Route
from pydantic import AnyHttpUrl
from mcp.server.auth.routes import create_protected_resource_routes
from mcp.server.auth.handlers.metadata import ProtectedResourceMetadataHandler
from mcp.shared.auth import ProtectedResourceMetadata
from mcp.server.auth.routes import cors_middleware

try:
    # Prefer FastMCP auth base classes when available
    from fastmcp.server.auth.auth import AuthProvider, AccessToken
except Exception:
    # Fallback stubs if fastmcp is unavailable at import time
    class OAuthProvider:  # type: ignore
        def verify_token(self, token: str):
            return None

    class AccessToken(dict):  # type: ignore
        pass

# Import configuration
from common.config import get_config

# Load configuration and set constants
_config = get_config()
ACCESS_TOKEN_EXPIRES_IN_SECONDS = _config.oauth.access_token_expires_in_seconds
AUTHORIZATION_CODE_EXPIRES_IN_SECONDS = _config.oauth.authorization_code_expires_in_seconds
ALLOWED_CALLBACK_URLS = _config.oauth.allowed_callback_urls
ALLOWED_CALLBACK_ORIGINS = _config.oauth.allowed_callback_origins
TRUSTED_CLIENT_IDS = _config.oauth.trusted_client_ids

class BvbrcOAuthProvider(AuthProvider):
    """
    Minimal custom OAuth provider for BV-BRC that implements the same behavioral logic
    as the prior module-level functions, with simple in-memory stores.

    It subclasses OAuthProvider for compatibility with FastMCP's auth interfaces.
    """

    def __init__(self, *, base_url: str, openid_config_url: str, authentication_url: str, allowed_callback_urls: list[str] | None = None) -> None:
        super().__init__(base_url=base_url, required_scopes=["profile", "token"])
        self.openid_config_url = openid_config_url
        self.authentication_url = authentication_url
        # Note: All localhost URLs (localhost, 127.0.0.1, any port) are automatically allowed
        # in addition to the URLs in this list
        # Use provided callback URLs or fall back to config
        config = get_config()
        self.allowed_callback_urls = allowed_callback_urls or config.oauth.allowed_callback_urls
        # Track issued tokens for validation (token -> username)
        self.issued_tokens: Dict[str, Dict[str, Any]] = {}

    # --- AuthProvider interface ---
    async def verify_token(self, token: str) -> AccessToken | None:  # type: ignore[override]
        """
        Verify token by checking it's one we issued through OAuth flow
        or a valid PATRIC token.
        """
        print(f"[TOKEN VERIFICATION] Verifying token: {token}", file=sys.stderr)
        
        if not token or not isinstance(token, str):
            print(f"[TOKEN VERIFICATION] Token is invalid or empty", file=sys.stderr)
            return None
        
        # Check if token is one we issued (stored when tokens are exchanged)
        token_info = self.issued_tokens.get(token)
        if not token_info:
            # Also check legacy storage in authorization_codes for backwards compatibility
            for auth_code_data in authorization_codes.values():
                if auth_code_data.get("user_token") == token:
                    token_info = {
                        "username": auth_code_data.get("username"),
                        "issued_at": auth_code_data.get("expires_at", time.time()) - AUTHORIZATION_CODE_EXPIRES_IN_SECONDS,
                    }
                    print(f"[TOKEN VERIFICATION] Token found in legacy storage. Token: {token}, Username: {token_info.get('username')}", file=sys.stderr)
                    break
        else:
            print(f"[TOKEN VERIFICATION] Token found in issued_tokens. Token: {token}, Username: {token_info.get('username')}", file=sys.stderr)
        
        # If not found in OAuth tokens, check if it's a PATRIC token
        if not token_info:
            # PATRIC tokens have format: un=username|tokenid=...|expiry=...|...
            if "un=" in token and "|tokenid=" in token:
                print(f"[TOKEN VERIFICATION] Detected PATRIC token format", file=sys.stderr)
                try:
                    # Parse PATRIC token
                    parts = token.split("|")
                    username = None
                    expiry = None
                    
                    for part in parts:
                        if part.startswith("un="):
                            username = part[3:]  # Remove "un=" prefix
                        elif part.startswith("expiry="):
                            expiry_str = part[7:]  # Remove "expiry=" prefix
                            try:
                                expiry = int(expiry_str)
                            except ValueError:
                                print(f"[TOKEN VERIFICATION] Invalid expiry format in PATRIC token", file=sys.stderr)
                                return None
                    
                    if not username:
                        print(f"[TOKEN VERIFICATION] Could not extract username from PATRIC token", file=sys.stderr)
                        return None
                    
                    # Check if token is expired
                    if expiry:
                        current_time = int(time.time())
                        if expiry < current_time:
                            print(f"[TOKEN VERIFICATION] PATRIC token expired. Expiry: {expiry}, Current: {current_time}", file=sys.stderr)
                            return None
                    
                    token_info = {
                        "username": username,
                        "issued_at": time.time() - ACCESS_TOKEN_EXPIRES_IN_SECONDS,  # Assume issued 1 hour ago
                    }
                    print(f"[TOKEN VERIFICATION] PATRIC token parsed successfully. Username: {username}, Expiry: {expiry}", file=sys.stderr)
                except Exception as e:
                    print(f"[TOKEN VERIFICATION] Exception parsing PATRIC token: {e}", file=sys.stderr)
                    return None
        
        if not token_info:
            # Token not found in our issued tokens and not a valid PATRIC token
            print(f"[TOKEN VERIFICATION] Token not found in issued tokens and not a valid PATRIC token. Token: {token}", file=sys.stderr)
            return None
        
        # Validate token is still valid by checking against authentication endpoint
        # Make a request to validate the token format/validity
        try:
            # Validate token by checking its format matches what the endpoint returns
            # The authentication endpoint returns tokens, so validate format
            if len(token.strip()) < 10:
                return None
            
            # Optional: Make a validation request to the authentication endpoint
            # Since it takes username/password, we can't directly validate tokens here
            # But we trust tokens we issued through our OAuth flow
            # If you need full validation, make a test API call to a BV-BRC endpoint that uses the token
            
        except Exception:
            print(f"[TOKEN VERIFICATION] Exception during validation. Token: {token}", file=sys.stderr)
            return None
        
        # Return an AccessToken as defined by mcp.server.auth.provider.AccessToken
        username = token_info.get("username", "unknown")
        print(f"[TOKEN VERIFICATION] Token verified successfully. Token: {token}, Username: {username}", file=sys.stderr)
        return AccessToken(
            token=token,
            client_id="bvbrc-public-client",
            scopes=["profile", "token"],
            expires_at=int(time.time()) + ACCESS_TOKEN_EXPIRES_IN_SECONDS,
        )

    # --- Helper methods ---
    def get_registered_client(self, client_id: str) -> dict | None:
        return self.registered_clients.get(client_id)

    # --- Instance wrappers for existing endpoint functions ---
    async def openid_configuration(self, request) -> JSONResponse:
        return openid_configuration(request, self.openid_config_url)

    async def oauth2_register(self, request) -> JSONResponse:
        return await oauth2_register(request)

    async def oauth2_authorize(self, request):
        return await oauth2_authorize(request, self.authentication_url)

    async def oauth2_login(self, request):
        return await oauth2_login(request, self.authentication_url)

    async def oauth2_token(self, request):
        return await oauth2_token(request, provider=self)

    def get_routes(self, mcp_path: str | None = "/mcp") -> list[Route]:
        routes: list[Route] = []
        if not self.base_url or not mcp_path:
            return routes
        # Build full resource URL and advertise protected resource metadata (RFC 9728)
        resource_url = f"{str(self.base_url).rstrip('/')}/{mcp_path.lstrip('/')}"
        # Authorization server URL is {base_url}/mcp since all OAuth endpoints are under /mcp
        authorization_server_url = AnyHttpUrl(resource_url)
        try:
            # Create PRM route for the specific resource path (/.well-known/oauth-protected-resource/mcp)
            routes.extend(
                create_protected_resource_routes(
                    resource_url=AnyHttpUrl(resource_url),
                    authorization_servers=[authorization_server_url],
                    scopes_supported=self.required_scopes,
                    resource_name="BV-BRC MCP",
                    resource_documentation=None,
                )
            )
            
            # Create PRM endpoint at /mcp/.well-known/oauth-protected-resource
            # This points to the Authorization Server for OAuth discovery
            # All OAuth paths should be under /mcp
            # Reuse the same authorization server URL
            root_metadata = ProtectedResourceMetadata(
                resource=AnyHttpUrl(str(self.base_url).rstrip('/')),
                authorization_servers=[authorization_server_url],
                scopes_supported=self.required_scopes,
                resource_name="BV-BRC MCP Server",
                resource_documentation=None,
            )
            root_handler = ProtectedResourceMetadataHandler(root_metadata)
            routes.append(
                Route(
                    "/mcp/.well-known/oauth-protected-resource",
                    endpoint=cors_middleware(root_handler.handle, ["GET", "OPTIONS"]),
                    methods=["GET", "OPTIONS"],
                )
            )
        except Exception as e:
            # If URL validation fails, skip advertising metadata to avoid breaking the app
            print(f"[AUTH] Failed to create PRM routes: {e}", file=sys.stderr)
            pass
        return routes

# Backward-compatible module-level stores for legacy function usage
registered_clients: Dict[str, Dict[str, Any]] = {}
authorization_codes: Dict[str, Dict[str, Any]] = {}

def is_localhost_url(url: str) -> bool:
    """
    Check if a URL is a localhost URL (localhost or 127.0.0.1, any port).
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        return hostname.lower() in ("localhost", "127.0.0.1", "::1") or hostname.startswith("127.")
    except Exception:
        return False

def is_allowed_redirect_uri(redirect_uri: str) -> bool:
    """
    Check if a redirect_uri is allowed by the configured allowlist.
    
    A redirect_uri is allowed if:
      1. It exactly matches an entry in ALLOWED_CALLBACK_URLS, OR
      2. Its origin (scheme + host) matches an entry in ALLOWED_CALLBACK_ORIGINS, OR
      3. It is a localhost URL.
    """
    # Exact match against the URL allowlist
    if redirect_uri in ALLOWED_CALLBACK_URLS:
        return True
    # Origin-based match: parse the redirect_uri and compare scheme+host
    # against each allowed origin. This supports clients (e.g. ChatGPT) that
    # use randomly-generated callback paths.
    try:
        parsed = urlparse(redirect_uri)
        redirect_origin = f"{parsed.scheme}://{parsed.hostname}".lower()
        for origin in ALLOWED_CALLBACK_ORIGINS:
            allowed = urlparse(origin)
            allowed_origin = f"{allowed.scheme}://{allowed.hostname}".lower()
            if redirect_origin == allowed_origin:
                return True
    except Exception:
        pass
    # Localhost bypass
    if is_localhost_url(redirect_uri):
        return True
    return False

def get_registered_client(client_id: str) -> dict | None:
    """Legacy helper used by module-level endpoints."""
    return registered_clients.get(client_id)

def can_auto_register(client_id: str) -> bool:
    """
    Check if a client_id is allowed to auto-register.
    
    Returns:
        True if auto-registration is allowed for this client_id
        False if auto-registration is disabled or client_id is not trusted
    """
    if TRUSTED_CLIENT_IDS is None:
        # Auto-registration disabled - must use /oauth2/register endpoint
        return False
    elif TRUSTED_CLIENT_IDS == []:
        # Empty list means allow all (current behavior)
        return True
    else:
        # Only allow if client_id is in trusted list
        return client_id in TRUSTED_CLIENT_IDS

def auto_register_client(client_id: str, redirect_uri: str, scope: str = "") -> dict:
    """
    Automatically register a client with the given client_id and redirect_uri.
    This is used when an unregistered client_id is provided during OAuth flow.
    
    Note: Only call this if can_auto_register(client_id) returns True.
    """
    client_data = {
        "client_id": client_id,
        "client_id_issued_at": int(time.time()),
        "redirect_uris": [redirect_uri],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code"],
        "response_types": ["code"],
        "client_name": client_id,  # Use client_id as default name
        "scope": scope,
    }
    
    registered_clients[client_id] = client_data
    print(f"Auto-registered client: {client_id} with redirect_uri: {redirect_uri}")
    print(f"Total registered clients: {len(registered_clients)}")
    
    return client_data

def openid_configuration(request, openid_config_url: str) -> JSONResponse:
    """
    Serves the OIDC discovery document that ChatGPT expects.
    """
    print("Query params:", dict(request.query_params))
    print("Request path:", request.url.path)
    config = {
            "issuer": openid_config_url,
            "authorization_endpoint": f"{openid_config_url}/mcp/oauth2/authorize",
            "token_endpoint": f"{openid_config_url}/mcp/oauth2/token",
            "registration_endpoint": f"{openid_config_url}/mcp/oauth2/register", # 1
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": ["profile", "token"],
            "claims_supported": ["sub", "api_token"]
    }
    return JSONResponse(content=config)

async def oauth2_register(request) -> JSONResponse:
    """
    Registers a new client with the OAuth2 server.
    Implements RFC 7591 OAuth 2.0 Dynamic Client Registration.
    """
    try:
        # Parse request body
        body = await request.json()
        print("Registration request body:", body)
        
        # Validate required fields
        if "redirect_uris" not in body or not body["redirect_uris"]:
            return JSONResponse(
                content={
                    "error": "invalid_client_metadata",
                    "error_description": "redirect_uris is required and must not be empty"
                },
                status_code=400
            )
        
        # Generate unique client_id
        client_id = str(uuid4())
        
        # Get auth method from request
        token_endpoint_auth_method = body.get("token_endpoint_auth_method", "none")
        
        # Generate client_secret if needed (not needed for "none" auth method)
        client_secret = None
        if token_endpoint_auth_method != "none":
            client_secret = secrets.token_hex(32)
        
        # Set timestamps
        client_id_issued_at = int(time.time())
        
        # Build response with client information from request
        response = {
            "client_id": client_id,
            "client_id_issued_at": client_id_issued_at,
            "redirect_uris": body["redirect_uris"],
            "token_endpoint_auth_method": token_endpoint_auth_method,
            "grant_types": body.get("grant_types", ["authorization_code", "refresh_token"]),
            "response_types": body.get("response_types", ["code"]),
        }
        
        # Add client_secret if generated
        if client_secret:
            response["client_secret"] = client_secret
            response["client_secret_expires_at"] = 0  # 0 means never expires
        
        # Add optional fields if provided in request
        optional_fields = [
            "client_name", "scope", "client_uri", "logo_uri", 
            "contacts", "tos_uri", "policy_uri", "jwks_uri", 
            "jwks", "software_id", "software_version"
        ]
        for field in optional_fields:
            if field in body:
                response[field] = body[field]
        
        # Store client information for later retrieval during authorization
        # NOTE: This module-level function remains for backward compatibility
        # when used directly; in class-based usage, the instance method is preferred.
        registered_clients[client_id] = response
        
        print(f"Registered new client: {client_id} ({body.get('client_name', 'unnamed')})")
        print(f"Total registered clients: {len(registered_clients)}")
        
        # Return 201 Created with client information
        return JSONResponse(content=response, status_code=201)
        
    except Exception as e:
        print(f"Registration error: {e}")
        return JSONResponse(
            content={
                "error": "server_error",
                "error_description": str(e)
            },
            status_code=500
        )

async def oauth2_authorize(request, authentication_url: str):
    """
    Authorization endpoint - displays login page for user authentication.
    This is where ChatGPT redirects the user to log in.
    """
    print("Authorization request received")
    print("Query params:", dict(request.query_params))
    
    # Extract OAuth2 parameters
    client_id = request.query_params.get("client_id")
    redirect_uri = request.query_params.get("redirect_uri")
    response_type = request.query_params.get("response_type")
    state = request.query_params.get("state")
    code_challenge = request.query_params.get("code_challenge")
    code_challenge_method = request.query_params.get("code_challenge_method")
    scope = request.query_params.get("scope", "")

    with open("images/bvbrc_logo_base64.txt", "r") as f:
        bvbrc_logo_base64 = f.read()
    
    # Validate required parameters
    if not client_id:
        return JSONResponse(
            content={"error": "invalid_request", "error_description": "client_id is required"},
            status_code=400
        )
    
    if not redirect_uri:
        return JSONResponse(
            content={"error": "invalid_request", "error_description": "redirect_uri is required"},
            status_code=400
        )
    
    if not response_type or response_type != "code":
        return JSONResponse(
            content={"error": "unsupported_response_type", "error_description": "Only 'code' response type is supported"},
            status_code=400
        )
    
    # Validate redirect_uri is whitelisted (allow localhost URLs, exact URL matches, or origin matches)
    if not is_allowed_redirect_uri(redirect_uri):
        return JSONResponse(
            content={
                "error": "invalid_request",
                "error_description": f"redirect_uri '{redirect_uri}' is not whitelisted. Allowed: {ALLOWED_CALLBACK_URLS} (exact), origins: {ALLOWED_CALLBACK_ORIGINS}, or any localhost URL"
            },
            status_code=400
        )
    
    # Check if client exists, if not, check if auto-registration is allowed
    client = get_registered_client(client_id)
    if not client:
        if can_auto_register(client_id):
            print(f"Client '{client_id}' not found, auto-registering...")
            client = auto_register_client(client_id, redirect_uri, scope)
        else:
            # Auto-registration not allowed for this client
            return JSONResponse(
                content={
                    "error": "invalid_client",
                    "error_description": f"Client '{client_id}' is not registered. Please register via /mcp/oauth2/register endpoint first."
                },
                status_code=400
            )
    
    # Validate redirect_uri matches registered URIs
    # If redirect_uri is not in the registered list, add it
    if redirect_uri not in client.get("redirect_uris", []):
        client["redirect_uris"].append(redirect_uri)
        print(f"Added redirect_uri '{redirect_uri}' to client '{client_id}' registered URIs")
    
    # Log client info (use client_name if available, otherwise just client_id)
    client_name = client.get('client_name', client_id) if client else client_id
    print(f"Authorizing client: {client_name}")
    print(f"Redirect URI: {redirect_uri}")
    print(f"Code challenge: {code_challenge}")
    
    # Serve login page
    login_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BV-BRC Login</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: #f5f5f5;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                padding: 20px;
            }}
            .login-container {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                padding: 40px;
                width: 100%;
                max-width: 420px;
            }}
            .logo {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .logo img {{
                max-width: 315px;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
            h1 {{
                color: #333;
                margin: 0 0 10px 0;
                font-size: 22px;
                text-align: center;
                font-weight: 600;
            }}
            .subtitle {{
                color: #666;
                text-align: center;
                margin-bottom: 30px;
                font-size: 14px;
            }}
            .client-info {{
                background: #f5f5f5;
                border-left: 4px solid #00567A;
                padding: 12px;
                margin-bottom: 25px;
                border-radius: 4px;
            }}
            .client-info p {{
                margin: 5px 0;
                font-size: 13px;
                color: #555;
            }}
            .client-info strong {{
                color: #333;
            }}
            .form-group {{
                margin-bottom: 20px;
            }}
            label {{
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
                font-size: 14px;
            }}
            input[type="text"],
            input[type="password"] {{
                width: 100%;
                padding: 12px;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                font-size: 14px;
                box-sizing: border-box;
                transition: border-color 0.3s;
                background: white;
                color: #333;
            }}
            input[type="text"]:focus,
            input[type="password"]:focus {{
                outline: none;
                border-color: #00567A;
            }}
            input[type="text"]::placeholder,
            input[type="password"]::placeholder {{
                color: #999;
            }}
            button {{
                width: 100%;
                padding: 14px;
                background: #00567A;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: background-color 0.2s, transform 0.1s;
                font-family: inherit;
            }}
            button:hover {{
                background: #004560;
                transform: translateY(-1px);
            }}
            button:active {{
                transform: translateY(0);
                background: #003d50;
            }}
            .error {{
                background: #fee;
                border-left: 4px solid #d32f2f;
                padding: 12px;
                margin-bottom: 20px;
                border-radius: 4px;
                color: #c33;
                font-size: 14px;
                display: none;
            }}
            .info {{
                text-align: center;
                margin-top: 20px;
                font-size: 12px;
                color: #666;
            }}
            .info a {{
                color: #1976d2;
                text-decoration: none;
            }}
            .info a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="logo">
                <img src="{bvbrc_logo_base64}" alt="BV-BRC Logo" />
            </div>
            <h1>Login</h1>
            <p class="subtitle">Authorize access to your BV-BRC MCP Resources</p>
            
            <div class="client-info">
                <p><strong>Application:</strong> {client.get('client_name', 'ChatGPT') if client else 'ChatGPT'}</p>
                <p><strong>Scopes:</strong> {scope or 'profile, token'}</p>
            </div>
            
            <div id="error-message" class="error"></div>
            
            <form id="login-form" method="POST" action="/mcp/oauth2/login">
                <input type="hidden" name="client_id" value="{client_id}">
                <input type="hidden" name="redirect_uri" value="{redirect_uri}">
                <input type="hidden" name="state" value="{state or ''}">
                <input type="hidden" name="code_challenge" value="{code_challenge or ''}">
                <input type="hidden" name="code_challenge_method" value="{code_challenge_method or ''}">
                <input type="hidden" name="scope" value="{scope}">
                
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required autofocus placeholder="Enter your BV-BRC username">
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required placeholder="Enter your password">
                </div>
                
                <button type="submit">Login & Authorize</button>
            </form>
            
            <p class="info">
                By logging in, you authorize this application to access your BV-BRC MCP Resources on your behalf.
            </p>
        </div>
        
        <script>
            document.getElementById('login-form').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const formData = new FormData(form);
                const errorDiv = document.getElementById('error-message');
                
                try {{
                    const response = await fetch(form.action, {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    if (response.redirected) {{
                        window.location.href = response.url;
                        return;
                    }}
                    
                    const data = await response.json();
                    
                    if (response.ok) {{
                        // Should not reach here as success should redirect
                        window.location.href = data.redirect_url;
                    }} else {{
                        errorDiv.textContent = data.error_description || data.error || 'Login failed';
                        errorDiv.style.display = 'block';
                    }}
                }} catch (error) {{
                    errorDiv.textContent = 'Network error. Please try again. ' + (error && error.message ? error.message : error);
                    errorDiv.style.display = 'block';
                }}
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=login_html)

async def oauth2_login(request, authentication_url: str):
    """
    Handles the login form submission.
    Authenticates the user and generates an authorization code.
    Redirects back to ChatGPT's callback URL with the code.
    """
    try:
        # Parse form data
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        client_id = form.get("client_id")
        redirect_uri = form.get("redirect_uri")
        state = form.get("state")
        code_challenge = form.get("code_challenge")
        code_challenge_method = form.get("code_challenge_method")
        scope = form.get("scope")
        
        print(f"Login attempt for user: {username}")
        
        # Validate required fields
        if not username or not password:
            return JSONResponse(
                content={"error": "invalid_request", "error_description": "Username and password are required"},
                status_code=400
            )
        
        # Authenticate user with BV-BRC API
        # POST request to authentication endpoint
        try:
            response = requests.post(
                authentication_url,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                data={
                    'username': username,
                    'password': password
                },
                timeout=30
            )
            
            # Check if authentication was successful
            if response.status_code != 200:
                print(f"Authentication failed for user {username}: HTTP {response.status_code}")
                return JSONResponse(
                    content={"error": "access_denied", "error_description": "Invalid username or password"},
                    status_code=401
                )
            
            # Parse the response to get the token
            # The token should be in the response body
            user_token = response.text.strip()
            
            if not user_token:
                raise ValueError("No token received from authentication endpoint")
            
            print(f"Login successful for user: {username}")
            print(f"Token received (first 20 chars): {user_token[:20]}...")
                
        except requests.RequestException as e:
            print(f"Authentication request failed for user {username}: {e}")
            return JSONResponse(
                content={"error": "server_error", "error_description": "Authentication service unavailable"},
                status_code=503
            )
        except Exception as e:
            print(f"Authentication failed for user {username}: {e}")
            return JSONResponse(
                content={"error": "access_denied", "error_description": "Authentication failed"},
                status_code=401
            )
        
        # Generate authorization code (one-time use, short-lived)
        auth_code = secrets.token_urlsafe(32)
        
        # Store authorization code with associated data
        # Code expires in 10 minutes
        authorization_codes[auth_code] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "scope": scope,
            "user_token": user_token,
            "username": username,
            "expires_at": time.time() + AUTHORIZATION_CODE_EXPIRES_IN_SECONDS,
            "used": False
        }
        
        print(f"Generated authorization code for user {username}")
        print(f"Total active auth codes: {len(authorization_codes)}")
        
        # Build redirect URL with authorization code
        params = {
            "code": auth_code
        }
        if state:
            params["state"] = state
        
        redirect_url = f"{redirect_uri}?{urlencode(params)}"
        
        print(f"Redirecting to: {redirect_url}")
        
        # Redirect user back to ChatGPT with the authorization code
        return RedirectResponse(url=redirect_url, status_code=302)
        
    except Exception as e:
        print(f"Login error: {e}")
        return JSONResponse(
            content={
                "error": "server_error",
                "error_description": str(e)
            },
            status_code=500
        )

async def oauth2_token(request, provider: Optional[Any] = None):
    """
    Handles the token request.
    Exchanges an authorization code for an access token.
    Retrieves the stored user token using the authorization code.
    """
    try:
        # Parse form data
        form = await request.form()
        code = form.get("code")
        client_id = form.get("client_id")
        client_secret = form.get("client_secret")
        redirect_uri = form.get("redirect_uri")
        grant_type = form.get("grant_type")
        code_verifier = form.get("code_verifier")  # For PKCE
        
        print(f"Token request received for code: {code[:20] if code else None}...")
        print(f"Client ID: {client_id}")

        if not code:
            return JSONResponse(
                content={"error": "invalid_request", "error_description": "code is required"},
                status_code=400
            )
        
        if not client_id:
            return JSONResponse(
                content={"error": "invalid_request", "error_description": "client_id is required"},
                status_code=400
            )
        
        if not redirect_uri:
            return JSONResponse(
                content={"error": "invalid_request", "error_description": "redirect_uri is required"},
                status_code=400
            )
        
        if not grant_type:
            return JSONResponse(
                content={"error": "invalid_request", "error_description": "grant_type is required"},
                status_code=400
            )
        
        if grant_type != "authorization_code":
            return JSONResponse(
                content={"error": "unsupported_grant_type", "error_description": "Only 'authorization_code' grant type is supported"},
                status_code=400
            )
        
        # Validate redirect_uri is whitelisted (allow localhost URLs, exact URL matches, or origin matches)
        if not is_allowed_redirect_uri(redirect_uri):
            return JSONResponse(
                content={
                    "error": "invalid_request",
                    "error_description": f"redirect_uri '{redirect_uri}' is not whitelisted. Allowed: {ALLOWED_CALLBACK_URLS} (exact), origins: {ALLOWED_CALLBACK_ORIGINS}, or any localhost URL"
                },
                status_code=400
            )
        
        # Check if client exists, if not, check if auto-registration is allowed
        client = get_registered_client(client_id)
        if not client:
            if can_auto_register(client_id):
                print(f"Client '{client_id}' not found during token exchange, auto-registering...")
                # Get scope from authorization code if available
                scope = ""
                if code and code in authorization_codes:
                    scope = authorization_codes[code].get("scope", "")
                client = auto_register_client(client_id, redirect_uri, scope)
            else:
                # Auto-registration not allowed for this client
                return JSONResponse(
                    content={
                        "error": "invalid_client",
                        "error_description": f"Client '{client_id}' is not registered. Please register via /mcp/oauth2/register endpoint first."
                    },
                    status_code=400
                )
        
        # Validate redirect_uri matches registered URIs
        # If redirect_uri is not in the registered list, add it
        if redirect_uri not in client.get("redirect_uris", []):
            client["redirect_uris"].append(redirect_uri)
            print(f"Added redirect_uri '{redirect_uri}' to client '{client_id}' registered URIs")
        
        # Validate authorization code exists
        if code not in authorization_codes:
            return JSONResponse(
                content={"error": "invalid_grant", "error_description": "Authorization code not found or invalid"},
                status_code=400
            )
        
        # Get stored authorization data
        auth_data = authorization_codes[code]
        
        # Validate authorization code is not used
        if auth_data["used"]:
            return JSONResponse(
                content={"error": "invalid_grant", "error_description": "Authorization code already used"},
                status_code=400
            )
        
        # Validate authorization code has not expired
        if auth_data["expires_at"] < time.time():
            return JSONResponse(
                content={"error": "invalid_grant", "error_description": "Authorization code expired"},
                status_code=400
            )
        
        # Validate client_id matches
        if auth_data["client_id"] != client_id:
            return JSONResponse(
                content={"error": "invalid_grant", "error_description": "Client ID mismatch"},
                status_code=400
            )
        
        # Validate redirect_uri matches
        if auth_data["redirect_uri"] != redirect_uri:
            return JSONResponse(
                content={"error": "invalid_grant", "error_description": "Redirect URI mismatch"},
                status_code=400
            )
        
        # Validate PKCE code verifier if code challenge was used
        if auth_data.get("code_challenge"):
            if not code_verifier:
                return JSONResponse(
                    content={"error": "invalid_grant", "error_description": "code_verifier is required for PKCE"},
                    status_code=400
                )
            
            # Verify code challenge (S256 method)
            computed_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).decode().rstrip('=')
            
            if computed_challenge != auth_data["code_challenge"]:
                return JSONResponse(
                    content={"error": "invalid_grant", "error_description": "Code verifier validation failed"},
                    status_code=400
                )
        
        # Mark authorization code as used
        auth_data["used"] = True
        
        # Retrieve the stored user token from the database (currently in-memory)
        user_token = auth_data["user_token"]
        stored_scope = auth_data.get("scope", "")
        username = auth_data.get("username")
        
        # Store token in provider for validation
        if provider:
            provider.issued_tokens[user_token] = {
                "username": username,
                "issued_at": time.time(),
            }
        
        print(f"Token exchange successful for user: {username}")
        print(f"Returning token (first 20 chars): {user_token[:20]}...")
        
        # Build token response with the stored user token
        token_response = {
            "access_token": user_token,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_EXPIRES_IN_SECONDS,
            "scope": stored_scope
        }

        return JSONResponse(content=token_response, status_code=200)
    except Exception as e:
        print(f"Token request error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={
                "error": "server_error",
                "error_description": str(e)
            },
            status_code=500
        )

