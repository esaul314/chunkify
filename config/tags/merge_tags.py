import yaml

# Load the original files
original_paths = {
    "generic": "./generic.yaml",
    "philosophy": "./philosophy.yaml",
    "technical": "./technical.yaml",
    "project_management": "./project_management.yaml",
    "psychology": "./psychology.yaml",
}

# Custom additions (from previous message)
custom_additions = {
    "generic": {
        "content_type": [
            "context_setting", "assumption_examination", "tension_resolution",
            "pattern_recognition", "misalignment_diagnosis"
        ],
        "complexity": ["ambiguous", "paradoxical"],
        "actionability": ["scaffolding", "sensemaking", "reflective_prompt"]
    },
    "philosophy": {
        "concepts": [
            "recognition", "selfhood", "intersubjectivity", "trauma",
            "developmental_morality", "systems_thinking"
        ],
        "philosophical_methods": [
            "dialogical", "developmental", "systems_theoretical",
            "second_order_observation"
        ]
    },
    "technical": {
        "concepts": [
            "observability", "feedback_loops", "coupling_and_cohesion",
            "technical_debt", "reliability", "operational_excellence"
        ],
        "technical_level": ["integration_pattern", "ops_maturity_model"]
    },
    "project_management": {
        "concepts": [
            "sensemaking", "emergence", "coordination", "psychological_safety",
            "co_creation", "feedback_mechanisms", "learning_organization"
        ],
        "phases": ["scaffolding", "divergence", "convergence"],
        "roles": ["facilitator", "bridge_role", "systemic_thinker"]
    },
    "psychology": {
        "concepts": [
            "inner_critics", "parts_work", "moral_development", "schema_defense",
            "identity_formation", "autonomy_and_belonging"
        ],
        "applications": ["coaching", "group_dynamics", "organizational_change"]
    }
}

# Merge utility
def merge_tags(original, additions):
    merged = original.copy()
    for key, values in additions.items():
        if key not in merged:
            merged[key] = []
        merged[key] = sorted(list(set(merged[key]) | set(values)))
    return merged

# Merge and write new YAMLs
updated_paths = {}
for name, path in original_paths.items():
    with open(path) as f:
        original_data = yaml.safe_load(f)

    updated_data = merge_tags(original_data, custom_additions[name])
    updated_path = f"./{name}_merged.yaml"
    with open(updated_path, "w") as f:
        yaml.dump(updated_data, f, sort_keys=False)
    updated_paths[name] = updated_path

updated_paths

