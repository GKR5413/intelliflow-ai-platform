output "bucket_names" {
  description = "Names of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.main : k => v.bucket }
}

output "bucket_arns" {
  description = "ARNs of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.main : k => v.arn }
}

output "bucket_domain_names" {
  description = "Domain names of the S3 buckets"
  value       = { for k, v in aws_s3_bucket.main : k => v.bucket_domain_name }
}
