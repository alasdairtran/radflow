# Maintenance

We document some useful code for maintenance. You probably don't need to worry
about this.

```sh
# Back up database
mkdir data/mongobackups
mongodump --db vevo --host=localhost --port=27017 --gzip --archive=data/mongobackups/vevo-2020-07-23.gz
mongodump --db wiki --host=localhost --port=27017 --gzip --archive=data/mongobackups/wiki-2020-07-23.gz

# Restore database
mongorestore --db vevo --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/vevo-2020-07-23.gz
mongorestore --db wiki --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/wiki-2020-07-23.gz
```
