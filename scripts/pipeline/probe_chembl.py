#!/usr/bin/env python3
import sys

import requests


def check_chembl_api():
    print("Probing ChEMBL API...")

    # 1. Check web status code for the API root
    try:
        r = requests.get("https://www.ebi.ac.uk/chembl/api/data/status", timeout=10)
        print(f"Status endpoint HTTP code: {r.status_code}")
        if r.status_code == 200:
            print(f"Content: {r.text.strip()}")
        else:
            print("API status endpoint returned non-200 status.")
    except Exception as e:
        print(f"Error connecting to status endpoint: {e}")

    # 2. Try a simple client request
    try:
        from chembl_webresource_client.new_client import new_client

        print(
            "Testing chembl_webresource_client with a simple query (Target: CHEMBL240)..."
        )
        target = new_client.target.filter(target_chembl_id="CHEMBL240").only(
            "target_chembl_id", "pref_name"
        )
        res = list(target)
        if res:
            print(f"Success! Found target: {res[0].get('pref_name')}")
            return True
        else:
            print("Query returned no results.")
            return False
    except Exception as e:
        print(f"Client request failed: {e}")
        return False


if __name__ == "__main__":
    if check_chembl_api():
        print("\nChEMBL API seems to be UP.")
        sys.exit(0)
    else:
        print("\nChEMBL API seems to be DOWN or unstable.")
        sys.exit(1)
