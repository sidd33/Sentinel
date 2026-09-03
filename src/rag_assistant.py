import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RiskPolicyRAG:
    """
    Grounded RAG Risk Verification Assistant.
    Retrieves defensive risk policy documents and provides analyst guidance.
    STRICT GUARDRAILS: Does NOT alter ML score, generate evasion techniques, or invent policies.
    """

    def __init__(self, policy_path="documents/risk_policies.json"):
        self.policy_path = policy_path
        self.policies = self._load_policies()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self._build_index()

    def _load_policies(self):
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"Policy document file missing at '{self.policy_path}'")
        with open(self.policy_path, "r") as f:
            return json.load(f)

    def _build_index(self):
        corpus = [
            f"{p['title']} {p['category']} {p['content']} {' '.join(p.get('rules', []))} {p['action']}"
            for p in self.policies
        ]
        return self.vectorizer.fit_transform(corpus)

    def retrieve(self, query, top_k=2):
        """
        Retrieves top_k relevant policy documents for a query.
        """
        # Guardrail check for offensive / bypass attempts
        prohibited_terms = ['evade', 'bypass', 'hack', 'spoof', 'generate fraud', 'exploit']
        if any(term in query.lower() for term in prohibited_terms):
            return [{
                'policy_id': 'GUARDRAIL_VIOLATION',
                'title': 'Defense-Only Protocol Enforcement',
                'category': 'Security Guardrail',
                'content': 'SentinelPay RAG Assistant operates strictly under defensive security guidelines. Requests asking to bypass detection, evade scoring, or generate fraud are rejected.',
                'action': 'Query rejected by defensive guardrail.',
                'score': 1.0
            }]

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                item = self.policies[idx].copy()
                item['relevance_score'] = float(similarities[idx])
                results.append(item)

        if not results:
            results.append(self.policies[0]) # Default fallback policy
        return results

    def verify_transaction_risk(self, ml_prob, locked_thresh, amount, customer_1d_tx, night_flag, terminal_spike_status="NORMAL"):
        """
        Generates a grounded risk verification explanation for a transaction based on ML predictions and retrieved policies.
        """
        is_fraud_predicted = ml_prob >= locked_thresh

        query_str = f"Transaction amount ${amount} customer tx count {customer_1d_tx} night flag {night_flag} terminal status {terminal_spike_status}"
        matched_policies = self.retrieve(query_str, top_k=2)

        explanation = f"### Risk Verification Summary (Authoritative ML Probability: {ml_prob*100:.1f}% | Locked Threshold: {locked_thresh*100:.1f}%)\n\n"
        explanation += f"**ML Decision**: {'🚨 HIGH FRAUD RISK' if is_fraud_predicted else '✅ APPROVED / LOW RISK'}\n\n"
        explanation += "#### Relevant Policy Citations:\n"

        for p in matched_policies:
            explanation += f"- **[{p['policy_id']}] {p['title']}** ({p['category']})\n"
            explanation += f"  - *Summary*: {p['content']}\n"
            explanation += f"  - *Recommended Action*: {p['action']}\n\n"

        explanation += "#### Analyst Investigation Recommendations:\n"
        if is_fraud_predicted:
            explanation += "1. Review customer historical transaction frequency (1-day & 7-day velocity).\n"
            explanation += "2. Verify terminal historical risk rate and recent payout spikes.\n"
            explanation += "3. Request 3D-Secure secondary authentication or place standard 24h settlement hold.\n"
        else:
            explanation += "1. Standard automated processing authorized.\n"
            explanation += "2. Continue real-time terminal risk monitoring.\n"

        return {
            'ml_probability': ml_prob,
            'locked_threshold': locked_thresh,
            'decision': 'REJECT/VERIFY' if is_fraud_predicted else 'APPROVE',
            'explanation_markdown': explanation,
            'cited_policies': [p.get('policy_id') for p in matched_policies]
        }

if __name__ == "__main__":
    rag = RiskPolicyRAG()
    res = rag.retrieve("How to respond to a merchant fraud spike?")
    print("RAG Retrieval Test:")
    for r in res:
        print(f" -> [{r['policy_id']}] {r['title']} (Score: {r.get('relevance_score', 0):.3f})")
    
    verif = rag.verify_transaction_risk(0.85, 0.45, 320.0, 1, 1, "ALERT_HIGH")
    print("\nTransaction Risk Verification Output:")
    print(verif['explanation_markdown'])
