#!/bin/bash

cat /etc/ssh/sshd_config_test_clion
/usr/sbin/sshd -D -e -f /etc/ssh/sshd_config_test_clion
service ssh restart
tail -f /dev/null