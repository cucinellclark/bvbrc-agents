"""
SRA Functions

This module provides functions for retrieving SRA (Sequence Read Archive) metadata.
"""

import json
import subprocess
import tempfile
import os
from typing import Dict, Any, List


def get_sra_metadata_func(sra_ids: List[str], container_path: str) -> Dict[str, Any]:
    """
    Retrieve metadata for one or more SRA IDs using p3-sra tool in a Singularity container.
    
    Args:
        sra_ids: List of SRA IDs to process (e.g., ["SRR37108646", "SRR37108647"])
        container_path: Path to the Singularity container file
    
    Returns:
        Dictionary with:
        - results: List of result objects, one per SRA ID
        - total_requested: Total number of SRA IDs requested
        - total_successful: Number of successfully processed IDs
        - total_failed: Number of failed IDs
    """
    # Validate inputs
    if not sra_ids:
        return {
            "error": True,
            "errorType": "INVALID_PARAMETERS",
            "message": "No SRA IDs provided",
            "source": "bvbrc-sra-tools"
        }
    
    if not container_path or not os.path.exists(container_path):
        return {
            "error": True,
            "errorType": "INVALID_PARAMETERS",
            "message": f"Singularity container not found: {container_path}",
            "source": "bvbrc-sra-tools"
        }
    
    results = []
    total_successful = 0
    total_failed = 0
    
    # Process each SRA ID
    for sra_id in sra_ids:
        result = {
            "sra_id": sra_id,
            "success": False,
            "metadata": None,
            "error": None
        }
        
        # Use a temporary directory for each SRA ID to avoid conflicts
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = os.path.join(temp_dir, "mdf")
            
            try:
                # Build the singularity command
                cmd = [
                    "singularity", "exec",
                    container_path,
                    "p3-sra",
                    "--id", sra_id,
                    "--metaonly",
                    "--out", temp_dir,
                    "--metadata-file", "mdf"
                ]
                
                # Execute the command
                process_result = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per SRA ID
                )
                
                # Check if command succeeded
                if process_result.returncode != 0:
                    result["error"] = f"Command failed with return code {process_result.returncode}. stderr: {process_result.stderr}"
                    results.append(result)
                    total_failed += 1
                    continue
                
                # Check if metadata file was created
                if not os.path.exists(metadata_file):
                    result["error"] = f"Metadata file not created. stdout: {process_result.stdout}, stderr: {process_result.stderr}"
                    results.append(result)
                    total_failed += 1
                    continue
                
                # Read and parse the metadata file
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_content = f.read()
                
                # Parse JSON metadata
                try:
                    metadata = json.loads(metadata_content)
                    # The metadata is a JSON array, extract the first element if it exists
                    if isinstance(metadata, list) and len(metadata) > 0:
                        result["metadata"] = metadata[0]
                    elif isinstance(metadata, dict):
                        result["metadata"] = metadata
                    else:
                        result["metadata"] = metadata
                    
                    result["success"] = True
                    total_successful += 1
                except json.JSONDecodeError as e:
                    result["error"] = f"Failed to parse JSON metadata: {str(e)}. Content: {metadata_content[:500]}"
                    results.append(result)
                    total_failed += 1
                    continue
                
            except subprocess.TimeoutExpired:
                result["error"] = "Command timed out after 5 minutes"
                results.append(result)
                total_failed += 1
                continue
            except Exception as e:
                result["error"] = f"Unexpected error: {str(e)}"
                results.append(result)
                total_failed += 1
                continue
        
        results.append(result)
    
    return {
        "results": results,
        "total_requested": len(sra_ids),
        "total_successful": total_successful,
        "total_failed": total_failed,
        "source": "bvbrc-sra-tools"
    }

