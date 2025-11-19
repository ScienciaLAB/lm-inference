import modal

# Connect to your existing Modal app
app = modal.App(name="docling-serve-gpu")

# Connect to the existing volume
artifacts_vol = modal.Volume.from_name("docling-artifacts")

# Local PDF path on your machine
local_file = "/Users/mandamac1/Downloads/scie1.pdf"

# Path inside the container volume
remote_path = "/scie1.pdf"  # this will be accessible in the container at /artifacts/scie1.pdf

# Upload the file to the volume
artifacts_vol.put(local_file, remote_path)

print(f"Uploaded {local_file} to the Modal volume at {remote_path}")
