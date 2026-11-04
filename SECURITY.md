# Security and deployment notes

This software is intended for defensive monitoring, research and controlled enterprise/lab validation.

- Do not capture traffic without authorization.
- Protect uploaded PCAPs and logs because they may contain sensitive network metadata.
- Run live capture with least privilege and isolate the collector.
- Do not connect automated blocking actions directly to the forecast engine without a separate policy/approval layer.
- Store secrets outside source control.
- Validate model performance and false-positive rates before operational use.
