output "cluster_endpoint" { value = aws_docdb_cluster.main.endpoint }
output "cluster_port" { value = aws_docdb_cluster.main.port }
output "cluster_master_username" { value = aws_docdb_cluster.main.master_username }
