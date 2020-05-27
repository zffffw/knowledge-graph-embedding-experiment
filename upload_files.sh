ssh-copy-id -p $1 -i ~/.ssh/id_rsa.pub $2
ssh -p $1 $2 "mkdir newKGE; mkdir newKGE/data; mkdir newKGE/checkpoint; mkdir newKGE/cluster"
scp -r -P $1 *  $2:~/newKGE/

