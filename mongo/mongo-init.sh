mongosh << EOF
use admin
db.auth("$MONGO_INITDB_ROOT_USERNAME", "$MONGO_INITDB_ROOT_PASSWORD")
use "$MONGO_INITDB_DATABASE" 
db.createCollection("deployments")
EOF