# Research-only boundaries and data protection

1. **Use only authorized de-identified CSVs**. No health data governance or
   legal/privacy compliance is implemented by this prototype.
2. **Never expose the default server to public traffic**. The frontend and API
   are designed for one trusted local user, not public clinical deployment.
3. HTTP training requires an explicit env opt-in plus an administrative secret.
   Even then, there is no user management, quotas, or persistent job queue.
4. Predict endpoints intentionally do not log patient predictors. HTTP access
   logs may include endpoints/statuses, not request bodies in standard setup.
5. Do not treat the result as a diagnostic recommendation or validated risk.
6. Before any research on identifiable health data, establish ethics approval,
   appropriate legal basis, storage policies and protected infrastructure.
7. The model file uses joblib/pickle. Only load trusted locally created artifacts.
8. An uploaded training CSV is parsed into memory; trained estimator and report
   persist on disk. Make sure your local artifact directory has safe permissions.
