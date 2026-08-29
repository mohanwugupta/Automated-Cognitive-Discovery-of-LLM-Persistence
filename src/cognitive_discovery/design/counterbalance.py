from cognitive_discovery.ontology.task_schema import ResponseMapping


def counterbalanced_mappings(labels=("X", "Y")) -> tuple[ResponseMapping, ResponseMapping]:
    labels = tuple(labels)
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError("counterbalancing requires two distinct labels")
    return ResponseMapping(labels[0], labels[1]), ResponseMapping(labels[1], labels[0])

