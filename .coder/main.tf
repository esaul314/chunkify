data "coder_workspace" "me" {}

resource "coder_agent" "main" {
  os   = "linux"
  arch = "amd64"

  startup_script = <<-EOT
    #!/bin/sh
    set -e

    # Install dependencies
    sudo apt-get update
    sudo apt-get install -y python3.11 python3-pip git

    # Install project dependencies
    python3.11 -m pip install --upgrade pip
    pip3 install -r requirements.txt
    pip3 install -e .[dev]

    # Run tests
    nox -s lint typecheck tests
  EOT
}
