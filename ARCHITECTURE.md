# DeepFace System Architecture

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Input Layer"
        A[Image Input<br/>File/URL/Array/Base64]
        B[Webcam Stream]
        C[Database Images]
    end
    
    subgraph "DeepFace Core API"
        D[DeepFace.verify]
        E[DeepFace.find]
        F[DeepFace.analyze]
        G[DeepFace.represent]
        H[DeepFace.stream]
        I[DeepFace.extract_faces]
    end
    
    subgraph "Processing Modules"
        J[Detection Module]
        K[Verification Module]
        L[Recognition Module]
        M[Demography Module]
        N[Representation Module]
        O[Streaming Module]
    end
    
    subgraph "Model Layer"
        P[Face Detectors<br/>10+ options]
        Q[Recognition Models<br/>11 models]
        R[Age Model]
        S[Gender Model]
        T[Emotion Model]
        U[Race Model]
        V[Anti-Spoofing Model]
    end
    
    subgraph "Output Layer"
        W[Verification Results]
        X[Recognition Matches]
        Y[Demographic Analysis]
        Z[Embeddings]
        AA[Real-time Display]
    end
    
    A --> D & E & F & G & I
    B --> H
    C --> E
    
    D --> K
    E --> L
    F --> M
    G --> N
    H --> O
    I --> J
    
    K --> J & Q
    L --> J & Q
    M --> J & R & S & T & U
    N --> J & Q
    O --> J & Q & R & S & T & U
    
    J --> P & V
    
    K --> W
    L --> X
    M --> Y
    N --> Z
    O --> AA
```

## Detailed Processing Pipeline

```mermaid
flowchart TD
    A[Input Image] --> B{Image Loading}
    B -->|Success| C[Image Preprocessing]
    B -->|Fail| ERR1[Error: Image Not Found]
    
    C --> D[Face Detection]
    D -->|Face Found| E[Face Alignment]
    D -->|No Face| ERR2{Enforce Detection?}
    ERR2 -->|Yes| ERR3[Error: Face Not Found]
    ERR2 -->|No| E
    
    E --> F[Anti-Spoofing Check]
    F -->|Real| G[Normalization]
    F -->|Fake| ERR4[Error: Spoof Detected]
    
    G --> H[Model-Specific<br/>Preprocessing]
    
    H --> I{Task Type}
    
    I -->|Recognition| J[Recognition Model]
    I -->|Demographics| K[Demographic Models]
    I -->|Both| L[Multiple Models]
    
    J --> M[Generate Embeddings]
    K --> N[Predict Attributes]
    L --> M & N
    
    M --> O{Operation}
    O -->|Verify| P[Calculate Distance]
    O -->|Find| Q[Search Database]
    O -->|Represent| R[Return Embeddings]
    
    N --> S[Return Demographics]
    
    P --> T[Compare with<br/>Threshold]
    Q --> U[Rank by Similarity]
    
    T --> V[Verification Result]
    U --> W[Recognition Results]
    R --> X[Embedding Vector]
    S --> Y[Age/Gender/<br/>Emotion/Race]
```

## Module Architecture

```mermaid
graph LR
    subgraph "deepface Package"
        A[DeepFace.py<br/>Main API]
        
        subgraph "modules/"
            B[detection.py]
            C[verification.py]
            D[recognition.py]
            E[representation.py]
            F[demography.py]
            G[streaming.py]
            H[modeling.py]
            I[preprocessing.py]
            J[normalization.py]
        end
        
        subgraph "models/"
            K[FacialRecognition]
            L[Detector]
            M[Demography]
            
            subgraph "facial_recognition/"
                N[VGGFace]
                O[Facenet]
                P[ArcFace]
                Q[DeepFace]
                R[OpenFace]
                S[DeepID]
                T[Dlib]
                U[SFace]
                V[GhostFaceNet]
                W[Buffalo_L]
            end
            
            subgraph "face_detection/"
                X[OpenCv]
                Y[Ssd]
                Z[Dlib]
                AA[MtCnn]
                AB[RetinaFace]
                AC[MediaPipe]
                AD[Yolo]
                AE[YuNet]
                AF[CenterFace]
            end
            
            subgraph "demography/"
                AG[Age]
                AH[Gender]
                AI[Emotion]
                AJ[Race]
            end
            
            subgraph "spoofing/"
                AK[FasNet]
            end
        end
        
        subgraph "commons/"
            AL[image_utils]
            AM[logger]
            AN[weight_utils]
            AO[folder_utils]
        end
        
        subgraph "config/"
            AP[threshold]
            AQ[confidence]
        end
        
        subgraph "database/"
            AR[postgres]
            AS[mongo]
            AT[weaviate]
        end
    end
    
    A --> B & C & D & E & F & G
    B & C & D & E & F & G --> H & I & J
    H --> K & L & M
    K --> N & O & P & Q & R & S & T & U & V & W
    L --> X & Y & Z & AA & AB & AC & AD & AE & AF
    M --> AG & AH & AI & AJ
    B --> AK
    B & C & D & E & F --> AL & AM & AP & AQ
    H --> AN & AO
    D --> AR & AS & AT
```

## Face Recognition Model Architecture

```mermaid
graph TB
    subgraph "Input Processing"
        A[Raw Image<br/>Any Size] --> B[Face Detection]
        B --> C[Crop & Align Face]
        C --> D[Resize to Model Input]
    end
    
    subgraph "Model-Specific Inputs"
        D --> E1[VGG-Face: 224x224]
        D --> E2[ArcFace: 112x112]
        D --> E3[FaceNet: 160x160]
    end
    
    subgraph "Deep Neural Networks"
        E1 --> F1[VGG-16<br/>Architecture]
        E2 --> F2[ResNet-34<br/>Architecture]
        E3 --> F3[Inception-ResNet<br/>Architecture]
    end
    
    subgraph "Embedding Layer"
        F1 --> G1[4096-d Vector]
        F2 --> G2[512-d Vector]
        F3 --> G3[128-d or 512-d Vector]
    end
    
    subgraph "Post-Processing"
        G1 & G2 & G3 --> H[L2 Normalization<br/>Optional]
        H --> I[Final Embeddings]
    end
    
    subgraph "Similarity Computation"
        I --> J1[Cosine Similarity]
        I --> J2[Euclidean Distance]
        I --> J3[Euclidean L2]
        I --> J4[Angular Distance]
    end
    
    J1 & J2 & J3 & J4 --> K[Distance/Similarity Score]
    K --> L{Compare with<br/>Threshold}
    L -->|< Threshold| M[Same Person]
    L -->|≥ Threshold| N[Different Person]
```

## Face Detection Pipeline

```mermaid
flowchart LR
    A[Input Image] --> B{Detector Type}
    
    B -->|Haar Cascade| C1[OpenCV<br/>Fast, Simple]
    B -->|Deep Learning| C2[SSD/YOLO<br/>Fast, Accurate]
    B -->|Cascade CNN| C3[MTCNN<br/>3-Stage Pipeline]
    B -->|State-of-Art| C4[RetinaFace<br/>Best Accuracy]
    
    C1 --> D1[Face Bounding Box]
    C2 --> D2[Face + Confidence]
    C3 --> D3[Face + 5 Landmarks]
    C4 --> D4[Face + 5 Landmarks<br/>+ High Confidence]
    
    D1 & D2 --> E{Alignment?}
    D3 & D4 --> F[Eye-based Alignment]
    
    E -->|Yes| G[Estimate Landmarks]
    E -->|No| H[Use Raw Detection]
    
    G --> F
    F --> I[Aligned Face Region]
    H --> I
    
    I --> J[Expand Face Region<br/>by Percentage]
    J --> K[Final Face Image]
```

## Database Search Architecture

```mermaid
graph TB
    subgraph "Input"
        A[Query Image] --> B[Extract Embedding]
    end
    
    subgraph "Database Types"
        C1[File System<br/>Pickle Cache]
        C2[PostgreSQL<br/>with pgvector]
        C3[MongoDB<br/>with vector search]
        C4[Weaviate<br/>Vector DB]
    end
    
    subgraph "Search Methods"
        B --> D{Search Type}
        D -->|Exact| E[Brute Force<br/>All Comparisons]
        D -->|ANN| F[Approximate<br/>Nearest Neighbor]
    end
    
    E --> C1
    F --> G{Backend}
    
    G -->|Faiss Index| C2
    G -->|Faiss Index| C3
    G -->|Native| C4
    
    subgraph "Scalability"
        C1 --> H1[~10K faces<br/><1 second]
        C2 & C3 --> H2[10K-1M faces<br/>seconds]
        C4 --> H3[1M-Billions<br/>seconds]
    end
    
    H1 & H2 & H3 --> I[Ranked Results]
    
    I --> J{Similarity Search?}
    J -->|No| K[Filter by Threshold<br/>Identity Match]
    J -->|Yes| L[Return Top-K<br/>Similar Faces]
    
    K & L --> M[Final Results]
```

## Real-Time Streaming Architecture

```mermaid
sequenceDiagram
    participant W as Webcam
    participant C as Capture Thread
    participant D as Detection
    participant B as Buffer
    participant P as Processing Thread
    participant M as Models
    participant UI as Display
    
    W->>C: Continuous Frame Feed
    C->>D: Check for Face
    
    alt Face Detected
        D->>B: Add to Buffer
        Note over B: Wait for 5 consecutive frames
        
        alt 5 Frames Ready
            B->>P: Trigger Processing
            P->>M: Run Recognition
            P->>M: Run Demographics
            M->>P: Return Results
            P->>UI: Update Display
            Note over UI: Show for 5 seconds
        end
    else No Face
        D->>C: Continue Capture
    end
    
    loop Every Frame
        C->>W: Get Next Frame
    end
```

## Data Flow for Verification

```mermaid
graph LR
    A[Image 1] --> B[Detect Face 1]
    C[Image 2] --> D[Detect Face 2]
    
    B --> E[Align Face 1]
    D --> F[Align Face 2]
    
    E --> G[Preprocess 1]
    F --> H[Preprocess 2]
    
    G --> I[Recognition Model]
    H --> I
    
    I --> J[Embedding 1<br/>512-d vector]
    I --> K[Embedding 2<br/>512-d vector]
    
    J & K --> L[Calculate Distance]
    
    L --> M{Distance Metric}
    M -->|Cosine| N[cos(θ)]
    M -->|Euclidean| O[||e1-e2||]
    M -->|Euclidean L2| P[||e1-e2||₂]
    
    N & O & P --> Q[Distance Value]
    
    Q --> R{< Threshold?}
    R -->|Yes| S[Same Person<br/>verified: true]
    R -->|No| T[Different Person<br/>verified: false]
    
    S & T --> U[Return JSON Result]
```

## Multi-Face Processing

```mermaid
flowchart TD
    A[Input Image] --> B[Face Detection]
    B --> C{Number of Faces}
    
    C -->|0 Faces| D{Enforce Detection?}
    D -->|Yes| E[Raise Exception]
    D -->|No| F[Return Empty List]
    
    C -->|1 Face| G[Process Single Face]
    C -->|Multiple Faces| H[Process All Faces]
    
    H --> I[Face 1]
    H --> J[Face 2]
    H --> K[Face N]
    
    G --> L[Generate Embedding]
    I --> M[Embedding 1]
    J --> N[Embedding 2]
    K --> O[Embedding N]
    
    L --> P[Single Result Dict]
    M & N & O --> Q[List of Result Dicts]
    
    P & Q --> R[Return Results]
```

## Security Features

```mermaid
graph TB
    subgraph "Anti-Spoofing Pipeline"
        A[Input Face] --> B[FasNet Model]
        B --> C{Analysis}
        C -->|Texture| D[Surface Analysis]
        C -->|Depth| E[3D Cues]
        C -->|Motion| F[Liveness]
        D & E & F --> G[Real/Fake Score]
        G --> H{Score > Threshold}
        H -->|Yes| I[Accept: Real Face]
        H -->|No| J[Reject: Spoofing Detected]
    end
    
    subgraph "Homomorphic Encryption"
        K[Original Embedding] --> L[LightPHE Encryption]
        L --> M[Encrypted Embedding]
        M --> N[Cloud Storage/Processing]
        N --> O[Distance Calculation<br/>on Encrypted Data]
        O --> P[Encrypted Result]
        P --> Q[On-Premise Decryption]
        Q --> R[Final Verification]
    end
```

## API Service Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        A[Web Browser]
        B[Mobile App]
        C[Desktop App]
        D[curl/Postman]
    end
    
    subgraph "API Gateway"
        E[Flask REST API<br/>Port 5005]
        F[CORS Middleware]
    end
    
    subgraph "API Routes"
        G[/verify]
        H[/find]
        I[/analyze]
        J[/represent]
        K[/register]
        L[/search]
    end
    
    subgraph "Core Services"
        M[DeepFace Core]
        N[Model Manager]
        O[Database Manager]
    end
    
    subgraph "Storage"
        P[File System]
        Q[PostgreSQL]
        R[MongoDB]
        S[Weaviate]
    end
    
    A & B & C & D --> E
    E --> F
    F --> G & H & I & J & K & L
    G & H & I & J --> M
    K & L --> O
    M --> N
    N --> P
    O --> Q & R & S
```

## Performance Optimization

```mermaid
graph LR
    subgraph "Speed Optimizations"
        A[Face Detection] --> B{Cache Strategy}
        B --> C[Pickle Cache<br/>for Embeddings]
        B --> D[Skip Detection<br/>if pre-cropped]
        
        E[Model Loading] --> F[Lazy Loading]
        F --> G[Load on First Use]
        
        H[Batch Processing] --> I[Process Multiple<br/>Images Together]
    end
    
    subgraph "Accuracy Optimizations"
        J[Better Detector] --> K[RetinaFace/MTCNN]
        L[Face Alignment] --> M[+6% Accuracy]
        N[Better Model] --> O[ArcFace/FaceNet512]
    end
    
    subgraph "Scalability"
        P[Small DB<br/>< 10K] --> Q[Exact Search<br/>Pickle Cache]
        R[Medium DB<br/>10K-1M] --> S[ANN with Faiss<br/>Postgres/Mongo]
        T[Large DB<br/>> 1M] --> U[Vector DB<br/>Weaviate]
    end
```

## Key Design Patterns

1. **Strategy Pattern**: Multiple interchangeable models for each task
2. **Factory Pattern**: Model creation and loading
3. **Facade Pattern**: Simple API hiding complex operations
4. **Cache Pattern**: Pickle files for embeddings
5. **Pipeline Pattern**: Sequential processing stages
6. **Observer Pattern**: Real-time streaming with callbacks

## Technology Stack

```mermaid
graph TB
    subgraph "Core Framework"
        A[Python 3.x]
        B[TensorFlow/Keras]
        C[NumPy]
    end
    
    subgraph "Computer Vision"
        D[OpenCV]
        E[PIL/Pillow]
    end
    
    subgraph "Web Service"
        F[Flask]
        G[Gunicorn]
        H[Flask-CORS]
    end
    
    subgraph "Database"
        I[psycopg2<br/>PostgreSQL]
        J[pymongo<br/>MongoDB]
        K[weaviate-client]
    end
    
    subgraph "ML/AI"
        L[MTCNN]
        M[RetinaFace]
        N[MediaPipe]
    end
    
    subgraph "Utilities"
        O[pandas]
        P[tqdm]
        Q[gdown]
    end
    
    subgraph "Security"
        R[LightPHE<br/>Encryption]
    end
```

