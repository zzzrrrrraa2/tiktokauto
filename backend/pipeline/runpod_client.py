import time
import json
import os
import requests
from typing import Optional


class RunPodClient:
    """Client for RunPod Serverless GPU endpoints."""

    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def submit_job(self, payload: dict) -> str:
        """Submit a job to the serverless endpoint. Returns job ID."""
        resp = requests.post(
            f"{self.base_url}/run",
            json={"input": payload},
            headers=self._headers(),
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data["id"]

    def get_job_status(self, job_id: str) -> dict:
        """Get job status. Returns {status, output, delay_time, ...}"""
        resp = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self._headers(),
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_job(self, job_id: str, poll_interval: float = 2.0, timeout: float = 600) -> dict:
        """Poll until job completes or timeout. Returns final status dict."""
        start = time.time()
        while True:
            status = self.get_job_status(job_id)
            if status["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                return status
            if time.time() - start > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
            time.sleep(poll_interval)

    def run_sync(self, payload: dict, timeout: float = 600) -> dict:
        """Submit and wait for completion. Returns output dict."""
        job_id = self.submit_job(payload)
        status = self.wait_for_job(job_id, timeout=timeout)
        if status["status"] != "COMPLETED":
            raise RuntimeError(f"Job {job_id} failed: {status}")
        return status["output"]

    def cancel_job(self, job_id: str):
        """Cancel a running job."""
        requests.post(
            f"{self.base_url}/cancel/{job_id}",
            headers=self._headers(),
            timeout=10
        )
