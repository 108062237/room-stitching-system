"""Starting room selection, insertion frontier and connection ordering."""


def rank_starting_rooms(scene):
    scores = {room: 0.0 for room in scene.polygons}
    for edge in scene.edges:
        if edge.manual_action != "layout_suggestion_pending":
            scores[edge.room_a] += edge.confidence
            scores[edge.room_b] += edge.confidence
    ranked = sorted(scores, key=lambda room: (-scores[room], room))
    if scene.root_id:
        ranked.remove(scene.root_id)
        ranked.insert(0, scene.root_id)
    return ranked, scores


def find_available_connections(edges, placed, excluded_pair_ids):
    result = []
    for edge in edges:
        pair_id = f"edge_{edge.index:04d}"
        if pair_id in excluded_pair_ids:
            continue
        a, b = edge.room_a in placed, edge.room_b in placed
        if a == b:
            continue
        parent, child = (edge.room_a, edge.room_b) if a else (edge.room_b, edge.room_a)
        result.append({"room_id": child, "connect_to": parent, "pair_id": pair_id,
                       "score": float(edge.confidence), "risk_tier": edge.risk_tier,
                       "risk_reasons": list(edge.risk_reasons),
                       "source": "da3_layout_suggestion" if edge.manual_action == "layout_suggestion_pending" else "geoverify_layout"})
    return result


def rank_connections(connections):
    # Break equal scores by room suffixes, preserving input order for exact ties.
    return sorted(connections, key=lambda item: (-item["score"], f"{item['connect_to'][-6:]} -> {item['room_id'][-6:]}"))
