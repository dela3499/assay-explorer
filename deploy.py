import sh

update_path = "/home/ubuntu/prod/assay-explorer/update_server.py"
server = sh.ssh.bake('-i','my-pair.pem','ubuntu@52.8.164.123')
server.python(update_path)