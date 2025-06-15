# 2025-summer-research-project

Our 2025 summer research project in ZJUI, which is about generative models.

the structure of our project:

```bash
/ 2025-summer-research-project
│
├── README.md         ← Project intro and instructions
├── .gitignore        ← Ignore build/temp files
├── /src              ← Source code
│   ├── main.py
│   └── ...
├── /docs             ← Documentation
├── /shared-resources ← learning 
├── /learning logs    ← recordings 
└── /tests            ← Test scripts

```

there are 3 branches.

|   Branch    | Purpose                             |
| :---------: | ----------------------------------- |
|   `main`    | Always working and deployable       |
|    `dev`    | Integration branch (testing)        |
| `feature/*` | New features/test (individual work) |
Some useful command for collaboration:

1.  Clone the repo

2.  Create a feature branch:

    ```bash
    
    git checkout -b feature/some-feature-name
    ```

3.  Do their work and push:

    ```bash
    
    git push -u origin feature/some-feature-name
    ```

4.  Go to GitHub and open a **Pull Request** from `feature/*` → `dev`



5.   check :Go to **Pull Requests** tab,Open the PR, read the code, and check.
6.   If good:Click **Merge pull request**
