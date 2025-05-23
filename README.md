# Talent or Luck? Evaluating Attribution Bias in Large Language Models

This work introduces a cognitively grounded benchmark to evaluate **attribution biases** in Large Language Models (LLMs) through the perspective of **Attribution Theory**. We examine how models assign causes to individuals' successes and failures, probing whether they favor internal (e.g., effort, ability) or external (e.g., luck, task ease) explanations, and whether these patterns differ across gender, race, religion, and nationality.

## ⚙️ Framework

Each scenario presents a success/failure event involving one or more social identities. The model is asked to select from four causal attributions:

- **High Effort** (internal)
- **High Ability** (internal)
- **Easy Task** (external)
- **Good Luck** (external)

Evaluations are conducted across:
- **Single-Actor** scenarios
- **Actor-Actor** comparisons
- **Actor-Observer)** scenarios

## 📁 Repo Structure

```bash
.
├── data/     
│   ├── closed_ended/ 
│   └── names/                    
├── figs/
│   └── closed_ended/
│       └── single_actor/       
│       └── actor_actor/       
│       └── actor_observer/       
├── src/    
│   ├── closed_ended/   
│   └── viz/                           
└── README.md                   
```

## 📂 Data Access

The full set of model outputs is available separately.

**Note:** Due to anonymity policies related to ongoing submissions, any large files (such as model outputs) will be shared **early through a non-anonymous link**. Public access links will be added here once anonymity is no longer required.
