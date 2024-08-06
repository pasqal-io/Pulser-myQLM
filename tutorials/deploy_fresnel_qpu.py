"""Example of connecting to a VM and deploying a MyQLM server using FresnelQPU."""

from pulser_myqlm import FresnelQPU

VM_IP = "127.0.0.1"
VM_PORT = 4300

# a FresnelQPU using pulser-simulation to simulate Jobs
dummy_fresnel = FresnelQPU(None)

# a FresnelQPU connected to a remote VM
print("Connecting to IP:", VM_IP, ", port:", VM_PORT)
fresnel_qpu = FresnelQPU(f"http://{VM_IP}:{VM_PORT}/api/", version="v1")
print("Connected. QPU is operational:", fresnel_qpu.is_operational)

# Deploy the QPU on a port and ip
SERVER_IP = "127.0.0.1"
SERVER_PORT = 1234
print("Creating a server on IP: ", SERVER_IP, ", PORT: ", SERVER_PORT)
fresnel_qpu.serve(SERVER_PORT, SERVER_IP)

# Connect to this server remotely with RemoteQPU(SERVER_PORT, SERVER_IP)
