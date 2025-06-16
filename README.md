# 2025-summer-research-project

Our 2025 summer research project in ZJUI, which is about flow-matching models.

The structure of our project:

```text
/ 2025-summer-research-project
│
├── README.md         ← Project intro and instructions
├── .gitignore        ← Ignore build/temp files
├── /src              ← Source code for our program
│   ├── main.py 
│   └── ...
├── /docs             ← Documentation for our program
├── /shared-resources ← learning resources(websites and books)
├── /learning-logs    ← recordings 
└── /tests            ← Test scripts
```
说明:

-   /src 存放的是大项目的源代码,/tests是调试产生的信息,属于第二阶段
-   /share-resource 包括我们学习进程中值得分享的网址(blog和网课)/PDF/数据集
-   /learning-logs 是每天自己的记录,学了什么,还有什么疑惑,相当于自我陈述反思
-   欢迎大家一起修改补充完善,我认为中英混杂挺好,看的懂就行.

+++++

There are 3 branches that forms our repository:

|   Branch    | Purpose                             |
| :---------: | ----------------------------------- |
|   `main`    | Always working and deployable       |
|    `dev`    | Integration branch (testing)        |
| `feature/*` | New features/test (individual work) 每个人都有 |

>    说明如何使用不同branch:
>
>   (我也不会)

+++++

### Some useful command for collaboration:

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
