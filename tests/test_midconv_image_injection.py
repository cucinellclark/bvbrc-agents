#!/usr/bin/env python3
"""
Test: Can vLLM on mango accept image_url content blocks in a user message
that appears MID-CONVERSATION (after assistant + tool messages)?

This simulates the exact message sequence that the agent loop would produce
when a tool like `view_workspace_image` returns image data:

    1. system message
    2. user message (text only -- the initial query)
    3. assistant message with tool_calls
    4. tool result message (text only -- metadata about the image)
    5. user message with image_url content blocks  <-- THIS IS THE KEY TEST
    6. LLM should produce a response describing the image

We test three scenarios:
    A. Mid-conversation image injection (the actual use case)
    B. Varying image sizes (64 KB, 256 KB, 512 KB, 1 MB, 2 MB, 4 MB base64)
    C. Multiple images in a single user message

Usage:
    python3 test_midconv_image_injection.py

Requires: openai package, network access to mango.cels.anl.gov:8004
"""

import base64
import io
import re
import struct
import sys
import time
import zlib
from openai import OpenAI

# -- Config --
BASE_URL = "http://mango.cels.anl.gov:8004/v1"
API_KEY = "not-needed"
MODEL = "Qwen/Qwen3.6-35B-A3B"
MAX_TOKENS = 4096


def create_test_png(width: int, height: int, color: tuple = (255, 0, 0)) -> bytes:
    """Create a minimal valid PNG image of given dimensions with a solid color.
    
    Returns raw PNG bytes (not base64).
    """
    # PNG header
    header = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    
    # IDAT chunk (image data)
    raw_data = b''
    row = bytes([0] + list(color) * width)  # filter byte + RGB pixels
    for _ in range(height):
        raw_data += row
    compressed = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xFFFFFFFF
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
    
    # IEND chunk
    iend_crc = zlib.crc32(b'IEND') & 0xFFFFFFFF
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    
    return header + ihdr + idat + iend


def create_large_png(target_raw_bytes: int) -> tuple[str, int]:
    """Create a PNG whose raw bytes are approximately target_raw_bytes.
    
    Uses a cycling color pattern that compresses well (solid-ish rows).
    Returns (data_uri, base64_size_bytes).
    """
    width = 256
    height = max(1, target_raw_bytes // (width * 3 + 1))
    
    # For large images with cycling patterns, deflate compression ratio is
    # typically close to 3:1 or better (the LZW-like pattern is very compressible)
    # We need raw_size = width * height * (3 bytes/pixel + 1 filter byte)
    # So: target_raw_bytes ≈ height * (width * 3 + 1)
    # But we want the compressed data to be closer to 1:1, so we use 
    # level 1 compression (fastest, least compression) for larger images.
    
    header = b'\x89PNG\r\n\x1a\n'
    
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    
    # Generate pixel data with a simple deterministic pattern
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # filter byte (None filter)
        for x in range(width):
            r = (x * 7 + y * 13) % 256
            g = (x * 11 + y * 3) % 256
            b = (x * 5 + y * 17) % 256
            raw_data.extend([r, g, b])
    
    raw_bytes = bytes(raw_data)
    compressed = zlib.compress(raw_bytes, 1)  # level 1 compression
    
    idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xFFFFFFFF
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
    
    iend_crc = zlib.crc32(b'IEND') & 0xFFFFFFFF
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    
    png_bytes = header + ihdr + idat + iend
    b64 = base64.b64encode(png_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{b64}"
    
    return data_uri, len(b64)


def run_test(client: OpenAI, test_name: str, messages: list) -> dict:
    """Run a single test case and return results."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    
    # Summarize messages
    for i, msg in enumerate(messages):
        role = msg["role"]
        if isinstance(msg.get("content"), list):
            parts = []
            for block in msg["content"]:
                if block["type"] == "text":
                    parts.append(f'text({len(block["text"])} chars)')
                elif block["type"] == "image_url":
                    url = block["image_url"]["url"]
                    size_kb = len(url) / 1024
                    parts.append(f'image({size_kb:.0f} KB b64)')
            content_desc = " + ".join(parts)
        elif msg.get("tool_calls"):
            content_desc = f"tool_calls: {[tc['function']['name'] for tc in msg['tool_calls']]}"
        elif msg.get("tool_call_id"):
            content_desc = f"tool_result for {msg['tool_call_id']} ({len(msg.get('content',''))} chars)"
        else:
            content_desc = f"text({len(str(msg.get('content','')))  } chars)"
        print(f"  [{i}] {role}: {content_desc}")
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        elapsed = time.time() - start
        
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        usage = response.usage
        
        # Strip thinking tags if present
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        print(f"\n  Status: SUCCESS")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Finish reason: {finish_reason}")
        print(f"  Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
        print(f"  Response preview: {clean_content[:300]}...")
        
        # Check if the model actually "saw" the image (describes visual content)
        image_aware = any(kw in clean_content.lower() for kw in [
            "red", "color", "image", "pixel", "pattern", "green",
            "blue", "gradient", "solid", "cyan", "magenta",
            "yellow", "picture", "visual", "chart", "square", "background",
            "cycle", "colorful", "hue", "spectrum", "uniform", "white",
            "black", "background", "foreground", "visible", "appears",
            "shows", "see", "display", "raster",
        ])
        print(f"  Image-aware response: {image_aware}")
        
        return {
            "test": test_name,
            "success": True,
            "elapsed": elapsed,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "image_aware": image_aware,
            "finish_reason": finish_reason,
            "content_preview": clean_content[:200],
        }
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  Status: FAILED")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Error: {type(e).__name__}: {e}")
        
        return {
            "test": test_name,
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
        }


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    # Verify model is available
    print(f"Connecting to {BASE_URL}, model={MODEL}")
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if MODEL not in available:
            print(f"ERROR: Model {MODEL} not available. Available: {available}")
            sys.exit(1)
        print(f"Model {MODEL} is available")
        for m in models.data:
            if m.id == MODEL:
                ctx = getattr(m, 'max_model_len', 'unknown')
                print(f"  Context window: {ctx} tokens")
    except Exception as e:
        print(f"ERROR: Cannot connect to {BASE_URL}: {e}")
        sys.exit(1)
    
    results = []
    
    # =========================================================================
    # TEST A: Mid-conversation image injection (the core use case)
    # =========================================================================
    tool_call_id = "call_test_001"
    
    # Create a recognizable test image: red square (so we can verify the model actually sees it)
    small_png = create_test_png(16, 16, (255, 0, 0))  # small red solid square
    small_b64 = base64.b64encode(small_png).decode()
    small_total_kb = len(small_b64) / 1024
    
    messages_a = [
        {"role": "system", "content": "You are a helpful assistant that can analyze images from workspace files."},
        {"role": "user", "content": "Please examine the image at /user/home/results/plot.png and describe what you see."},
        {"role": "assistant", "content": None, "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "view_workspace_image",
                    "arguments": '{"path": "/user/home/results/plot.png"}',
                },
            }
        ]},
        {"role": "tool", "tool_call_id": tool_call_id, "content": '{"filename": "plot.png", "mime_type": "image/png", "size_bytes": 1234, "path": "/user/home/results/plot.png"}'},
        # THIS IS THE KEY: a user message with image content blocks AFTER the tool result
        {"role": "user", "content": [
            {"type": "text", "text": "[System: The requested workspace image is shown below for your visual analysis.]"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{small_b64}"}},
        ]},
    ]
    results.append(run_test(client, f"A: Mid-conversation image injection (test {int(small_total_kb*100)} bytes, solid red square)", messages_a))
    
    # =========================================================================
    # TEST B: Image size scaling
    # =========================================================================
    sizes_raw_bytes = [64 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]  # target raw bytes
    
    for raw_target in sizes_raw_bytes:
        kb_label = raw_target // 1024
        data_uri, actual_b64_size = create_large_png(raw_target)
        actual_kb = actual_b64_size / 1024
        
        print(f"\n  [Prep] Raw target: {kb_label}KB, compressed image: ~{int(raw_target/3)} bytes raw, base64 output: {actual_kb:.0f} KB")
        
        messages_b = [
            {"role": "system", "content": "You are a helpful assistant that can analyze images. Describe any image you see in detail."},
            {"role": "user", "content": "Describe the image at /user/home/output.png and its visual properties."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_size_{kb_label}", "type": "function", "function": {
                    "name": "view_workspace_image",
                    "arguments": '{"path": "/user/home/output.png"}',
                }}
            ]},
            {"role": "tool", "tool_call_id": f"call_size_{kb_label}", "content": f'{{"filename": "output.png", "mime_type": "image/png", "size_bytes": {raw_target}}}'},
            {"role": "user", "content": [
                {"type": "text", "text": "[System: The requested workspace image is shown below for your visual analysis.]"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ]
        results.append(run_test(
            client, 
            f"B: Image size ~{kb_label} KB raw (base64: {actual_kb:.0f} KB)",
            messages_b
        ))
    
    # Also test 2 MB and 3 MB raw for the upper boundary
    for raw_target in [2 * 1024 * 1024, 3 * 1024 * 1024]:
        mb = raw_target / (1024 * 1024)
        try:
            data_uri, actual_b64_size = create_large_png(raw_target)
            actual_kb = actual_b64_size / 1024
            print(f"\n  [Prep] Raw target: {mb:.1f}MB, base64 output: {actual_kb:.0f} KB")
        except MemoryError:
            print(f"\n  [Prep] SKIPPED {mb:.1f}MB: too large to create locally")
            continue
        
        messages_b = [
            {"role": "system", "content": "You are a helpful assistant that can analyze images. Describe any image you see in detail."},
            {"role": "user", "content": "Describe the image at /user/home/output.png."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_size_{raw_target}", "type": "function", "function": {
                    "name": "view_workspace_image",
                    "arguments": '{"path": "/user/home/output.png"}',
                }}
            ]},
            {"role": "tool", "tool_call_id": f"call_size_{raw_target}", "content": f'{{"filename": "output.png", "size_bytes": {raw_target}}}'},
            {"role": "user", "content": [
                {"type": "text", "text": "[System: The requested workspace image is shown below.]"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]},
        ]
        results.append(run_test(
            client, 
            f"B: Edge case ~{mb:.1f}MB raw (base64: {actual_kb:.0f} KB)",
            messages_b
        ))
    
    # =========================================================================
    # TEST C: Multiple images in one injection
    # =========================================================================
    img1 = create_test_png(32, 32, (255, 0, 0))   # red
    img2 = create_test_png(32, 32, (0, 0, 0))     # black
    uri1 = f"data:image/png;base64,{base64.b64encode(img1).decode()}"
    uri2 = f"data:image/png;base64,{base64.b64encode(img2).decode()}"
    
    messages_c = [
        {"role": "system", "content": "You are a helpful assistant that can analyze images."},
        {"role": "user", "content": "Show me the two comparison images from /user/home/charts/"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_multi_1", "type": "function", "function": {
                "name": "view_workspace_image",
                "arguments": '{"path": "/user/home/charts/left.png"}',
            }},
            {"id": "call_multi_2", "type": "function", "function": {
                "name": "view_workspace_image",
                "arguments": '{"path": "/user/home/charts/right.png"}',
            }},
        ]},
        {"role": "tool", "tool_call_id": "call_multi_1", "content": '{"filename": "left.png", "size_bytes": 500}'},
        {"role": "tool", "tool_call_id": "call_multi_2", "content": '{"filename": "right.png", "size_bytes": 500}'},
        {"role": "user", "content": [
            {"type": "text", "text": "[System: The 2 requested workspace images are shown below for your visual analysis.]"},
            {"type": "image_url", "image_url": {"url": uri1}},
            {"type": "image_url", "image_url": {"url": uri2}},
        ]},
    ]
    results.append(run_test(client, "C: Multiple images in one injection (2 small images)", messages_c))
    
    # =========================================================================
    # TEST D: Verify it FAILS the other way (tool role with image_url)
    # =========================================================================
    # Just to confirm the limitation we're working around
    messages_d = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Describe this image: /user/home/output.png"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_fails", "type": "function", "function": {
                "name": "view_workspace_image",
                "arguments": '{"path": "/user/home/output.png"}',
            }},
        ]},
        {"role": "tool", "tool_call_id": "call_fails", "content": [
            {"type": "text", "text": '{"filename": "output.png", "size_bytes": 500}'},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{small_b64}"}},
        ]},
    ]
    results.append(run_test(client, "D: Control - tool role with image_url (expected to fail)", messages_d))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        extra = ""
        if r["success"]:
            extra = f" | {r['elapsed']:.1f}s | {r['prompt_tokens']} prompt tokens | image-aware: {r['image_aware']} | finish: {r['finish_reason']}"
        else:
            extra = f" | ERROR: {r.get('error', 'unknown')[:80]}"
        icon = "+" if r["image_aware"] else "o"
        print(f"  [{status}] {icon} {r['test']}{extra}")
    
    # Overall verdict
    core_passed = results[0]["success"] and results[0].get("image_aware", False)
    all_passed = all(r["success"] for r in results)
    
    print(f"\n  Core use case (mid-conv injection works + model sees image): {'PASS' if core_passed else 'FAIL'}")
    print(f"  All tests passed: {'PASS' if all_passed else 'FAIL'} ({sum(1 for r in results if r['success'])}/{len(results)})")
    
    if not core_passed:
        if not results[0]["success"]:
            print("\n  VERDICT: Mid-conversation image injection does NOT work with this model/endpoint.")
            print("  The OpenAI API rejected the message sequence. Need to try a different approach.")
        else:
            print("\n  VERDICT: Mid-conversation image injection works, but the model did not")
            print("  actually perceive the image (response was not image-aware). Might need larger image.")
    else:
        # Find max working size
        size_results = [r for r in results if r["test"].startswith("B:")]
        max_working_kb = 0
        for r in size_results:
            if r["success"] and r.get("image_aware"):
                import re
                m = re.search(r'~(\d+) KB', r["test"]) or re.search(r'(~[\d.]+)(MB)', r["test"])
                if m:
                    if m.group(2) == 'MB':
                        max_working_kb = max(max_working_kb, int(float(m.group(1)) * 1024))
                    else:
                        max_working_kb = max(max_working_kb, int(m.group(1)))
        
        print(f"\n  VERDICT: Mid-conversation image injection works!")
        if max_working_kb:
            print(f"  Max tested working image size: ~{max_working_kb} KB")
    
    sys.exit(0 if core_passed else 1)


if __name__ == "__main__":
    main()
