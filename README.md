# WildScan-AI ARCHITECTURE
project/
│
├── app/
│   ├── api/                # API Layer
│   │   └── routes.py
│   │
│   ├── application/        # Application Layer
│   │   └── inference.py
│   │
│   ├── domain/             # Business Layer
│   │   └── model_service.py
│   │
│   ├── infrastructure/     # External concerns
│   │   ├── model_loader.py
│   │   └── image_utils.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   └── main.py             # Entry point
│
├── model/
│   └── animal_model.pth
│
├── requirements.txt
└── README.md
