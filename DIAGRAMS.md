# Trust Graph + Diffusion Pipeline Diagrams

```mermaid
flowchart LR
    subgraph UI["User input"]
        U1["Product URL"]
        U2["Product API call"]

        U1 --> U2
    end

    subgraph ARG["ARGUS Core"]
        subgraph EW["eWOM Agent"]
            C0["Product Reviews"]
            C1["Sentiment Analysis"] 
            C2["Review Trust Score"]

            C3["eWOM Agent?"]

            C0 --> C1
            C0 --> C2
            C1 --> C3
            C2 --> C3
        end

        U2 --> |"Reviews"| C0

        subgraph TR["Trust Agent"]
            B0["Any Text"]
            B1["LLM Trust Factors (7 continuous scores)"]
            B2["Diffusion Score"]
            B3["Bayesian Network"]

            B0 --> B1
            B0 --> B2
            B1 --> B3
            B2 --> B3

        end


        C2 --- |"< Review Trust Score"| B3
        B0 --- |"< Review Text"| C2 
        U2 --> B0

        subgraph VA["Value Agent"]
            A0["JSON Product Schema"]
            A1["Retrieval by word TF-IDF + SVD"]
            A2["Retrieval by character TF-IDF + SVD"]
            A3["Weighted Average by similarity score"]

            A0 --> A1
            A0 --> A2
            A1 --> |"Top K"| A3
            A2 --> |"Top K"| A3

        end
    D["Monotonic BN Decision"]
    end

    C3 --> |"Score + Justification"| D
    B3 --> |"Score + Justification"| D
    A3 --> |"Score + Justification"| D


    F["User-facing Output: Score + Justification"]

    D --> F
    U2 --> A0


    %% Node color scheme
classDef input fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D47A1;
classDef ewom fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100;
classDef trust fill:#FCE4EC,stroke:#D81B60,stroke-width:2px,color:#880E4F;
classDef value fill:#EDE7F6,stroke:#5E35B1,stroke-width:2px,color:#311B92;
classDef decision fill:#FFF8E1,stroke:#F9A825,stroke-width:3px,color:#5D4037;
classDef output fill:#ECEFF1,stroke:#546E7A,stroke-width:2px,color:#263238;

%% Apply node classes
class U1,U2 input;
class C0,C1,C2,C3 ewom;
class B0,B1,B2,B3 trust;
class A0,A1,A2,A3 value;
class D decision;
class F output;

%% Subgraph frame + header text color
%% (requires subgraph IDs like: subgraph UI["User input"])
style UI fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#1E88E5
style ARG fill:#FAFAFA,stroke:#616161,stroke-width:2px,color:#616161
style EW fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#FB8C00
style TR fill:#FCE4EC,stroke:#D81B60,stroke-width:2px,color:#D81B60
style VA fill:#EDE7F6,stroke:#5E35B1,stroke-width:2px,color:#5E35B1

%% Arrow/edge colors by source component stroke
linkStyle 0,5,12,21 stroke:#1E88E5,stroke-width:2px,color:#1E88E5
linkStyle 1,2,3,4,10,17 stroke:#FB8C00,stroke-width:2px,color:#FB8C00
linkStyle 6,7,8,9,11,18 stroke:#D81B60,stroke-width:2px,color:#D81B60
linkStyle 13,14,15,16,19 stroke:#5E35B1,stroke-width:2px,color:#5E35B1
linkStyle 20 stroke:#F9A825,stroke-width:2px,color:#F9A825

```




## 1) Training and Artifact Build Flow

```mermaid
flowchart LR
    subgraph UI["User input"]
        U1["Product URL"]
        U2["Product API call"]

        U1 --> U2
    end

    U2 --> |"Reviews"| C0

    subgraph ARG["ARGUS sub-agents"]
        subgraph EW["eWOM Agent"]
            C0["Product Reviews"]
            C1["Sentiment Analysis"]
            C2["Review Trust Score"]
            C3["eWOM Agent?"]
            C0 --> C1
            C0 --> C2
            C1 --> C3
            C2 --> C3
        end


        subgraph TR["Trust Agent"]
            B0["Any Text"]
            B1["LLM Trust Factors (7 continuous scores)"]
            B2["Diffusion Score"]
            B3["Bayesian Network"]
            B0 --> B1
            B0 --> B2
            B1 --> B3
            B2 --> B3
        end

        C2 --> B0
        B0 --> C2
        U2 --> B0
        U2 --> A0

        subgraph VA["Value Agent"]
            A0["JSON Product Schema"]
            A1["Retrieval by word TF-IDF + SVD"]
            A2["Retrieval by character TF-IDF + SVD"]
            A3["Weighted Average by similarity score"]
            A0 --> A1
            A0 --> A2
            A1 -->|"Top K"| A3
            A2 -->|"Top K"| A3
        end
    end

    D["Monotonic BN Decision"]
    C3 -->|"Score + Justification"| D
    B3 -->|"Score + Justification"| D
    A3 -->|"Score + Justification"| D

    F["User-facing Output: Trust Score + Justification"]
    D --> F
    



```

```mermaid
flowchart LR
    A["Fake Reviews Dataset"] --> B["Phase A Sampling (target_rows=240)"]
    B --> C["LLM Labeling (7 trust factors)"]
    C --> D["Trust Features + Buckets"]

    D --> E["BN Baseline Training"]
    D --> F["Logistic Baseline Training"]

    B --> G["Diffusion Text Model Training<br/>TF-IDF -> SVD -> Scaler<br/>Forward Corruption + Reverse Denoiser + Classifier"]
    G --> H["diffusion_real_score / diffusion_bucket"]

    H --> I["BN + Diffusion Factor Training"]
    H --> J["Logistic + Diffusion Factor Training"]
    D --> I
    D --> J

    E --> K["graph_model.json"]
    F --> L["logistic_model.json"]
    I --> M["graph_model_with_diffusion.json"]
    J --> N["logistic_model_with_diffusion.json"]
    G --> O["diffusion_fork_model.joblib"]
```

## 2) Deploy Runtime Mode Selection and Inference

```mermaid
flowchart TD
    A["Input Products / Text"] --> B["validate_environment()"]
    B --> C{"Trust artifacts present?"}

    C -->|Yes| D["Mode: trust_graph"]
    C -->|No| E{"Standalone diffusion bundle present?"}
    E -->|Yes| F["Mode: standalone_diffusion"]
    E -->|No| G["Return environment error"]

    D --> H["Normalize product text"]
    H --> I["Label via Ollama (with cache)"]
    I --> J["Base scoring:<br/>BN + Logistic"]
    J --> K{"Diffusion fork artifacts complete?"}
    K -->|Yes| L["Add diffusion scoring:<br/>diffusion_real_score + augmented BN/Logistic"]
    K -->|No| M["Return base trust scores only"]
    L --> N["Standard deploy JSON output"]
    M --> N

    F --> P["Normalize raw text rows"]
    P --> Q["Diffusion bundle inference"]
    Q --> R["Output p_real / p_fake / prediction_std"]
```

## 3) Scoring Composition (Before vs After Diffusion Fork)

```mermaid
flowchart LR
    A["LLM Trust Factors (7 continuous scores)"] --> B["Bucketization"]
    A --> C["Logistic Baseline"]
    B --> D["BN Baseline"]

    E["Diffusion Text Signal<br/>diffusion_real_score"] --> F["diffusion_bucket"]
    E --> G["Logistic + Diffusion"]
    F --> H["BN + Diffusion"]

    A --> G
    B --> H

    D --> I["phase_b_truth_likelihood_graph"]
    C --> J["phase_b_truth_likelihood_logistic"]
    H --> K["phase_b_truth_likelihood_graph_with_diffusion"]
    G --> L["phase_b_truth_likelihood_logistic_with_diffusion"]

    I --> M["trust_risk_index_graph = 1 - graph_prob"]
    J --> N["trust_risk_index_logistic = 1 - logistic_prob"]
    K --> O["trust_risk_index_graph_with_diffusion"]
    L --> P["trust_risk_index_logistic_with_diffusion"]
```


```mermaid
flowchart TD
    A["Input labeled_df
    (text, label_truth, trust scores/buckets)"] --> B["Train/Test split
    (stratified by label_truth)"]

    B --> C1["Baseline branch (no diffusion)
Train BN on trust buckets
Train LR on trust scores"]
    C1 --> C2["Baseline test predictions
p_bn_baseline, p_logistic_baseline"]

    B --> D0["Diffusion fork: TRAIN LOOP"]

    subgraph TRAIN_LOOP["Diffusion model training"]
      D0 --> D1["Train texts -> TF-IDF -> SVD -> Scaler
x_train (latent vectors)"]
      D1 --> D2["Build diffusion schedule (betas, alpha_bar)"]
      D2 --> D3["Forward corruption samples
(x_t, eps_true, t) from x_train"]
      D3 --> D4["Train denoiser (Ridge)
input: [x_t, t] -> target: eps_true"]
      D4 --> D5["Create classifier train data
x0_hat = denoise(x_t, t)
x_cls = [x_train ; x0_hat]
y_cls = [y_train ; y_train_repeated]"]
      D5 --> D6["Train diffusion classifier (LogisticRegression)
on x_cls, y_cls"]
    end

    D6 --> E0["Diffusion fork: VALIDATION/TEST LOOP"]

    subgraph VAL_LOOP["Validation/Test inference"]
      E0 --> E1["Test texts -> TF-IDF/SVD/Scaler
x_test"]
      E1 --> E2["Repeat k=1..N (inference_samples):
corrupt x_test -> x_t
denoise -> x0_hat
classifier -> p_real_k"]
      E2 --> E3["Aggregate:
diffusion_real_score = mean(p_real_k)
diffusion_prediction_std = std(p_real_k)
diffusion_bucket = bucket(diffusion_real_score)"]
    end

    E3 --> F1["Augment trust features:
+ diffusion_real_score (LR)
+ diffusion_bucket (BN)"]
    F1 --> F2["Train/evaluate augmented models:
BN_with_diffusion
LR_with_diffusion"]
    C2 --> G["Compare baseline vs augmented metrics"]
    F2 --> G

    G --> H["Outputs:
phase_a_bn_diffusion_fork_metrics.csv
phase_a_bn_diffusion_fork_test_predictions.csv
graph_model_with_diffusion.json
logistic_model_with_diffusion.json
diffusion_fork_model.joblib"]

```