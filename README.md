# semantic_segmentation_template

### 1. Install requirements

`pip install -r requirements.txt` 

### 2. Train and Validate

```python
/mnt/home/miniconda3/bin/python trainval.py -e pascal_all -sb ../results -r 1
```

Argument Descriptions:
```
-e  [Experiment group to run like 'vae' (the rest of the experiment groups are in exp_configs/main_exps.py)] 
-sb [Directory where the experiments are saved]
-r  [Flag for whether to reset the experiments]
-d  [Directory where the datasets are aved]
```

or with vscode

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "pascal_all",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/trainval.py",
            "console": "integratedTerminal",
            "args": [
                "-e",
                "pascal_all",
                "-r",
                "1",
                "-d",
                "/mnt/public/datasets",
                "-sb",
                "/mnt/public/results/debug/active_learning",
                "-nw",
                "8"
            ],
        },
    ]
}
```

### 3. Visualize the Results

Follow these steps to visualize plots. Open `results.ipynb`, run the first cell to get a dashboard like in the gif below, click on the "plots" tab, then click on "Display plots" or the "images" tab and then click on "Display Images". Parameters of the plots can be adjusted in the dashboard for custom visualizations.

<p align="center" width="100%">
<img width="100%" src="https://raw.githubusercontent.com/haven-ai/haven-ai/master/docs/vis.gif">
</p>


