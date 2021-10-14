import os

if os.environ['EAI_ACCOUNT_ID'] == "7c445bd5-8d23-48dd-963e-b0480d238c8a":
    user = "pau"
elif os.environ['EAI_ACCOUNT_ID'] == "68fdc833-1eaa-4b60-9b64-ac27814e4b61": 
    user = 'sai'
else:
    user = 'issam'

JOB_CONFIG =  {'account_id':os.environ['EAI_ACCOUNT_ID'] ,
            'image': 'registry.console.elementai.com/snow.colab/ssh',
            'data': [
                        'snow.colab.public:/mnt/public',
                        f'snow.{user}.home:/mnt/home'
                        ],
            'restartable':True,
            'resources': {
                'cpu': 16,
                'mem': 64,
                'gpu': 1,
                'gpu_mem': 20,
                'gpu_model':'!A100'
            },
            'interactive': False,
            'bid': 0,
            }