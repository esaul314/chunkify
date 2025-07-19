# Let's generate refined and carefully reconsidered YAML versions based on Alex's interests and preferences.

generic_yaml = """
content_type:
  - context_setting
  - assumption_examination
  - tension_resolution
  - pattern_recognition
  - misalignment_diagnosis
  - exploratory
  - analytical
  - practical

complexity:
  - simple
  - intermediate
  - advanced
  - ambiguous
  - paradoxical

actionability:
  - informational
  - actionable
  - strategic
  - scaffolding
  - sensemaking
  - reflective_prompt
"""

philosophy_yaml = """
concepts:
  - ethics
  - epistemology
  - ontology
  - phenomenology
  - hermeneutics
  - existentialism
  - recognition
  - selfhood
  - intersubjectivity
  - trauma
  - developmental_morality
  - systems_thinking

philosophical_methods:
  - analytical
  - dialectical
  - phenomenological
  - pragmatic
  - dialogical
  - developmental
  - systems_theoretical
  - second_order_observation
"""

technical_yaml = """
concepts:
  - observability
  - monitoring
  - reliability
  - scalability
  - architecture
  - software_design
  - infrastructure_as_code
  - feedback_loops
  - coupling_and_cohesion
  - technical_debt
  - operational_excellence

technical_level:
  - foundational
  - intermediate
  - advanced
  - integration_pattern
  - ops_maturity_model
"""

project_management_yaml = """
concepts:
  - agile
  - waterfall
  - kanban
  - scrum
  - lean
  - sensemaking
  - emergence
  - coordination
  - psychological_safety
  - co_creation
  - feedback_mechanisms
  - learning_organization

phases:
  - initiation
  - planning
  - execution
  - monitoring
  - closure
  - scaffolding
  - divergence
  - convergence

roles:
  - manager
  - analyst
  - coordinator
  - lead
  - facilitator
  - bridge_role
  - systemic_thinker
"""

psychology_yaml = """
concepts:
  - trauma
  - resilience
  - motivation
  - cognitive_bias
  - emotional_intelligence
  - inner_critics
  - parts_work
  - moral_development
  - schema_defense
  - identity_formation
  - autonomy_and_belonging

applications:
  - therapy
  - education
  - workplace
  - coaching
  - group_dynamics
  - organizational_change
"""

# Save these refined YAML files
yaml_files = {
    "./generic_updated.yaml": generic_yaml,
    "./philosophy_updated.yaml": philosophy_yaml,
    "./technical_updated.yaml": technical_yaml,
    "./project_management_updated.yaml": project_management_yaml,
    "./psychology_updated.yaml": psychology_yaml,
}

# Writing the YAML files
for path, content in yaml_files.items():
    with open(path, "w") as file:
        file.write(content)

yaml_files.keys()

