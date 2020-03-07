ssh-copy-id -p $1 -i ~/.ssh/id_rsa.pub $2
scp -r -P $1 *  $2:~/newKGE/

