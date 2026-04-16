Get-Content "src/gary/sql/001_init.sql" | docker exec -i gary-postgres psql -U gary -d gary_db
