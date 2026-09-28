"""Role/modality checks after the unchanged legacy quote extractor.

These conservative lexical checks are not a semantic truth model. Quotes are
selected verbatim; an unsupported field stays empty with an explicit reason.
"""

import re

from app.ml.text import evidence_card, sentences, _photonic_sentence_signal, OPTICAL


OUTLOOK = re.compile(r"\b(?:may|might|could|would|should|potential\w*|promis\w*|prospect\w*|"
                     r"will|shall|anticipat\w*|foresee\w*|future|envision\w*|expect\w*|pav(?:e\w*|ing)|"
                     r"aim\w*|towards|opens?\s+(?:a\s+)?(?:path|way))\b", re.I)
REVIEW = re.compile(r"\b(?:we|this\s+(?:paper|work|article|review)|here|then)\b.{0,90}"
                    r"\b(?:review|survey|summari[sz]e|discuss|introduce|overview)\w*\b|"
                    r"\b(?:this\s+review|in\s+this\s+review|review\s+of|perspective|roadmap|point\s+of\s+view|"
                    r"organized\s+as|paper\s+is\s+organized)\b|"
                    r"\b(?:are|is)\s+(?:also\s+)?(?:discussed|reviewed|outlined|highlighted|summari[sz]ed)\b", re.I)
SIMULATION = re.compile(r"\b(?:simulat\w*|numerical(?:ly)?|theoretical(?:ly)?|in\s+silico)\b", re.I)
RESULT = re.compile(r"\b(?:measur\w*|experiment\w*|demonstrat\w*|achiev\w*|observ\w*|"
                    r"fabricat(?:e[sd]?|ing)|implement(?:s|ed|ing)?|"
                    r"reduc(?:e[sd]?|ing)|improv(?:e[sd]?|ing)|outperform\w*|enhanc(?:e[sd]?|ing))\b", re.I)
PROPOSAL = re.compile(r"\b(?:propos(?:e[sd]?|ing|als?)|conceptual|design\s+of|we\s+design)\b", re.I)
PROBLEM = re.compile(r"\b(?:bottleneck\w*|difficult\w*|suffer\w*|"
                     r"hinder\w*|restrict\w*|cannot|unable|lack\w*|drawbacks?)\b|"
                     r"\blimited\s+(?:by|to|in)\b|\blimitations\s+(?:of|in|due\s+to)\b|"
                     r"\b(?:(?:major|technical|key|remaining|fundamental)\s+)?challenges?\s+(?:include|is|are|of|in|for|to)\b|"
                     r"\b(?:is|are|remains?|be|very|highly)\s+(?:\w+\s+){0,2}challenging\b|"
                     r"\b(?:is|are|remains?)\s+(?:(?:largely|mostly|still|practically)\s+){0,2}"
                     r"(?:unexplored|unrealized|unavailable)\b", re.I)
CASE = re.compile(r"\b(?:we|here|this\s+(?:work|paper|study))\b.{0,110}\b"
                  r"(?:demonstrat(?:e[sd]?|ing)|propos(?:e[sd]?|ing)|present(?:s|ed|ing)?|"
                  r"fabricat(?:e[sd]?|ing)|construct(?:s|ed|ing)?|implement(?:s|ed|ing)?|"
                  r"develop(?:s|ed|ing)?|experiment(?:s|ed|ing)?|simulat(?:e[sd]?|ing)|design(?:s|ed|ing)?)\b", re.I)
LATEX = re.compile(r"\\(?:[a-zA-Z]+|[{}])|\$[^$]+\$")
DEMAND = re.compile(r"\b(?:needs?|demands?|requirements?)\s+(?:for|to)\b|"
                    r"\b(?:growing|increas\w*)\b.{0,60}\b(?:demands?|needs?)\b|"
                    r"\b(?:important|essential|crucial|necessary)\s+(?:for|to)\b|"
                    r"\bof\s+(?:(?:key|great|critical|major|fundamental)\s+)?importance\s+(?:for|to)\b|"
                    r"\b(?:targeting|targeted\s+at|aiming\s+at)\b|"
                    r"\b(?:need|needs)\s+to\s+be\b", re.I)
BENEFIT_DETAIL = re.compile(r"\b(?:efficien\w*|power|energy|latency|speed\w*|accuracy|precision|rates?|"
                            r"bandwidth|memory|densit\w*|cost\w*|noise|fidelity|stabil\w*|scalab\w*|"
                            r"programmab\w*|parallel\w*|robust\w*|reliab\w*|throughput|capacity|"
                            r"complexity|time|errors?|recognition|control\w*|integration|compact\w*|"
                            r"simpl\w*|nonlinear|activations?|inference|neurons?|readout|versatil\w*)\b", re.I)
BENEFIT_ASSERTION = re.compile(r"\b(?:reduces?|reduced|improves?|improved|enhances?|"
                               r"achieves?|achieved|outperforms?|provides?|enables?|offers?|"
                               r"avoids?|eliminates?|results?|shows?|showed|demonstrat\w*)\b", re.I)
SOLUTION_STATEMENT = re.compile(
    r"\b(?:address\w*|overcom\w*)\s+(?:(?:the|these|those|such|key|major)\s+)*"
    r"(?:challenges|limitations)\b|\bchallenges\s+(?:are|were|have\s+been)\s+"
    r"(?:\w+\s+){0,2}(?:overcome|addressed|resolved|solved)\b", re.I)


def _quote_context(text, study):
    parts = sentences(study["abstract"])
    indexes = [i for i, part in enumerate(parts) if part == text]
    if not indexes:
        return []
    index = indexes[0]
    return parts[max(0, index - 1):index] + parts[index + 1:index + 2]


def quote_assessment(field, quote, study):
    text = quote["text"]
    if quote["mode"] == "study_title":
        return {"accepted": field == "example", "modality": "research_reference",
                "reason": "bibliographic_example_not_an_execution_claim"}
    context = _quote_context(text, study)
    experimental = re.compile(r"\b(?:experimentally|fabricat(?:e[sd]?|ing)|measured|measurements|sequencing\s+results|in\s+vitro|"
                              r"wet(?:[ -]lab)?\s+experiments?(?:\s+results)?|"
                              r"(?:carried\s+out|conducted|performed)\s+(?:\w+[ -]+){0,4}experiments?|"
                              r"experimental\s+(?:results|validation)|construct(?:ed)?\b.{0,70}\b(?:devices?|transistors?))\b", re.I)
    numerical = re.compile(r"\b(?:numerical(?:ly)?|simulations?|simulated|theoretical(?:ly)?|in\s+silico)\b", re.I)
    own_numerical = re.search(r"\b(?:we|our|this\s+(?:work|paper|study))\b.{0,100}\b"
                              r"(?:numerically|simulate(?:d)?|theoretically|simulations?)\b", study["abstract"], re.I)
    simulation_context = (numerical.search(text) or (field != "problem" and not experimental.search(text) and
                          (any(numerical.search(part) for part in context) or
                           (own_numerical and not experimental.search(study["abstract"])))))
    if REVIEW.search(text):
        modality = "review"
    elif field == "problem":
        # A risk or limitation is not an experimental result. Words such as
        # 'might', 'theoretical phase' and 'simulation software' describe the
        # constraint here, not the method used to establish a benefit.
        modality = "source_statement"
    elif OUTLOOK.search(text):
        modality = "outlook"
    elif numerical.search(text) and experimental.search(text):
        modality = "simulation_and_experiment"
    elif simulation_context:
        modality = "simulation"
    elif experimental.search(text):
        modality = "reported_result"
    elif PROPOSAL.search(text) and not re.search(
            r"\b(?:we\s+(?:also\s+)?demonstrate|was\s+demonstrated|"
            r"and\s+(?:successfully\s+)?demonstrat(?:e[sd]?)|achieved|measured|observed)\b", text, re.I):
        modality = "proposal"
    elif re.search(r"\bcan\b", text, re.I):
        modality = "outlook" if re.search(r"\b(?:dramatic\w*|revolution\w*)\b", text, re.I) else "source_statement"
    elif RESULT.search(text):
        modality = "reported_result"
    elif PROPOSAL.search(text):
        modality = "proposal"
    else:
        modality = "source_statement"
    accepted, reason = True, "source_role_supported"
    if re.search(r"\b(?:high resolution image|download ms powerpoint|download powerpoint|view larger image)\b", text, re.I):
        accepted, reason = False, "publisher_interface_fragment"
    elif re.search(r"(?:\.{3,}|…)\s*[\"'’”]?\s*$", text):
        accepted, reason = False, "truncated_source_excerpt"
    elif field == "problem" and (not PROBLEM.search(text) or modality == "review"):
        accepted, reason = False, "not_a_problem_statement"
    elif field == "advantage" and (modality in {"review", "outlook", "proposal"} or DEMAND.search(text)):
        accepted, reason = False, "benefit_not_reported_as_a_result_or_property"
    elif field == "advantage" and not BENEFIT_DETAIL.search(text):
        accepted, reason = False, "benefit_not_specified"
    elif field == "advantage" and (PROPOSAL.search(text) or re.search(r"\b(?:study|explore|investigate|examine)\b", text, re.I)) and not BENEFIT_ASSERTION.search(text):
        accepted, reason = False, "research_goal_not_a_reported_benefit"
    elif field == "example" and (not CASE.search(text) or modality in {"review", "outlook"}):
        accepted, reason = False, "not_a_specific_research_example"
    if accepted and field == "problem":
        solution = SOLUTION_STATEMENT.search(text)
        if solution and solution.start() <= PROBLEM.search(text).start():
            accepted, reason = False, "solution_instead_of_constraint"
    if accepted and field == "advantage":
        if PROBLEM.search(text) and not BENEFIT_ASSERTION.search(text):
            accepted, reason = False, "constraint_instead_of_benefit"
        if re.search(r"\b(?:in hopes of|hope to|seeks? to|intends? to)\b", text, re.I):
            accepted, reason = False, "desired_benefit_not_reported"
        if re.search(r"\bcan\s+(?:\w+\s+){0,2}be\s+(?:\w+\s+){0,2}achieved\b", text, re.I):
            accepted, reason = False, "possible_benefit_not_established"
        if re.search(r"\bcontribut\w*\s+to\s+(?:(?:the|a|broader|better|deeper|our)\s+)*understanding\b|"
                     r"\bachiev\w*\s+(?:(?:significant|major|important|substantial)\s+)?breakthroughs?\b|"
                     r"\battract\w*\b.{0,35}\battention\b|"
                     r"\b(?:reviv\w*|renew\w*|spark\w*)\s+(?:research\s+)?interest\b", text, re.I):
            accepted, reason = False, "research_progress_instead_of_specific_benefit"
        if re.search(r"\bgap\s+between\b", text, re.I) and not re.search(r"\b(?:reduc\w*|narrow\w*|clos\w*|bridg\w*)\b", text, re.I):
            accepted, reason = False, "limitation_instead_of_benefit"
        purpose = re.search(r"\bto\s+(?:improve|reduce|enhance|achieve|enable|provide|avoid)\b", text, re.I)
        observed = re.search(r"\b(?:results?\s+(?:show|demonstrate)|experimentally|numerically|measured|"
                             r"achieved|outperformed|provides|enables|offers|reduces|improves|enhances)\b", text, re.I)
        if purpose and not observed:
            accepted, reason = False, "desired_benefit_not_reported"
    execution = study.get("execution", {})
    if accepted and field == "advantage" and execution:
        bound = any(text in evidence["text"] for evidence in execution.get("evidence", []))
        reference = re.match(r"^(?:additionally,?\s+|furthermore,?\s+|further,?\s+)?"
                             r"(?:our|this|these|the\s+(?:\w+[ -]+){0,8}"
                             r"(?:device|architecture|system|approach|method|configuration|network|scheme)|it)\b", text, re.I)
        contextual = reference and (OPTICAL.search(study["title"]) or any(OPTICAL.search(part) for part in context) or
                                    (execution.get("evidence_level") == "direct" and not re.search(
                                        r"\b(?:cpu|gpu|human\s+brain|digital\s+signal\s+processing)\b", text, re.I)))
        if not (OPTICAL.search(text) or bound or contextual):
            accepted, reason = False, "benefit_not_linked_to_photonic_implementation"
    if accepted and field == "example" and execution.get("evidence_level") == "direct":
        bound = any(text in evidence["text"] for evidence in execution.get("evidence", []))
        if not bound and not _photonic_sentence_signal(text, require_execution=True):
            accepted, reason = False, "example_does_not_identify_computational_execution"
    # A review can report prior results, but it does not become the developers' experiment.
    if accepted and study.get("type") == "review" and modality == "reported_result":
        modality = "reviewed_result"
    return {"accepted": accepted, "modality": modality, "reason": reason,
            "context": context, "modality_uses_context": bool(modality == "simulation" and simulation_context and not SIMULATION.search(text))}


def supported_card(studies, proposed, topic_terms=None):
    """Validate existing choices, then seek other verbatim excerpts of shown sources."""
    by_id = {s["id"]: s for s in studies[:12]}
    options = {field: [] for field in ("problem", "advantage", "example")}
    rejected = {field: [] for field in options}
    seen = set()

    def consider(field, quote, study, order):
        if quote is None:
            return
        key = (field, quote["study_id"], quote["text"], quote["mode"])
        if key in seen:
            return
        seen.add(key)
        assessment = quote_assessment(field, quote, study)
        if (assessment["accepted"] and field == "example" and topic_terms
                and not any(term.casefold() in study["title"].casefold() for term in topic_terms)):
            assessment.update(accepted=False, reason="example_does_not_identify_cluster_technology")
        if not assessment["accepted"]:
            rejected[field].append({"study_id": study["id"], "text": quote["text"], **assessment})
            return
        # Prefer a paper with direct execution to a merely nominal architecture.
        nominal = study.get("execution", {}).get("evidence_level") == "nominal"
        reference = quote["mode"] == "study_title"
        review = study.get("type") == "review" or assessment["modality"] == "reviewed_result"
        options[field].append(((reference, nominal, review, order, abs(len(quote["text"]) - 250), quote["text"]),
                               quote, assessment))

    for field, quote in proposed.items():
        if quote and quote["study_id"] in by_id:
            study = by_id[quote["study_id"]]
            consider(field, quote, study, list(by_id).index(study["id"]))
    for order, study in enumerate(studies[:12]):
        consider("example", {"text": study["title"], "study_id": study["id"], "url": study["url"],
                             "title": study["title"], "mode": "study_title"}, study, order)
        for passage in sentences(study["abstract"]):
            # The protected extractor still applies its original negation checks.
            extracted = evidence_card([{**study, "abstract": passage}])
            for field, quote in extracted.items():
                consider(field, quote, study, order)
    card, annotations, explanations = {}, {}, {}
    for field, items in options.items():
        if items:
            _, card[field], annotations[field] = min(items, key=lambda item: item[0])
            modality = annotations[field]["modality"]
            prefix = {"problem": "Ограничение, указанное в источнике.",
                      "advantage": "Преимущество, заявленное в источнике; границы утверждения сохранены в цитате.",
                      "example": "Пример исследования из доступного корпуса."}[field]
            qualifier = {"simulation": " В выбранном фрагменте приведён результат расчёта или симуляции.",
                         "simulation_and_experiment": " Источник сообщает и о симуляции, и о физическом эксперименте.",
                         "proposal": " В цитате описан предлагаемый подход.",
                         "reviewed_result": " Сведения приведены в обзоре других исследований.",
                         "research_reference": " Библиографическая ссылка; реализацию нельзя заключить из одного названия."}.get(modality, "")
            explanations[field] = prefix + qualifier
        else:
            card[field] = None
            annotations[field] = {"accepted": False, "modality": "not_found", "reason": "no_supported_excerpt"}
            explanations[field] = "Подходящий подтверждающий фрагмент в отобранных источниках не найден."
        annotations[field]["rejected_count"] = len(rejected[field])
    return card, annotations, explanations, rejected
