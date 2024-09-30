#!/bin/bash
chmod +x setup-inbound-dev.sh traffic-cloud.sh traffic-edge.sh
sudo ./setup-inbound-dev.sh
sudo DEV=eth0 ./traffic-edge.sh
sudo DEV=ifb0 ./traffic-edge.sh