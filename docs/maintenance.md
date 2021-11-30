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

# Compress experiment and data directories
tar -zcvf expt.tar.gz expt
tar -zcvf data.tar.gz data

# Upload to Nectar Containers
swift upload radflow data.tar.gz  --info -S 1073741824
swift upload radflow expt.tar.gz  --info -S 1073741824
```
