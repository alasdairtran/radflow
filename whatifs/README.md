# Demo

```sh
# Start only the zero container (don't start alpha yet)
docker-compose up zero

# The bulk loader takes 50 minutes to load 600M edges
docker exec -it $ZERO_CONTAINER_ID dgraph bulk \
    -f /backups/wiki.rdf \
    -s /backups/schema.graphql \
    --map_shards=1 \
    --reduce_shards=1 \
    --http localhost:8000 \
    --zero localhost:5080

# Shut down zero and start whole server
docker-compose up

# If we need to live load later
docker exec -it $RATEL_CONTAINER_ID dgraph live -f backups/wiki.rdf --alpha alpha:9080 --zero zero:5080 -c 1
```
