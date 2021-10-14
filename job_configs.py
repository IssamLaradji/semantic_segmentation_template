import os

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
