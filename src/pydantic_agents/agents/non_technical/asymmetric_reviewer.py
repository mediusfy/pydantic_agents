from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_agents.agents.non_technical._common import GROUNDING
from pydantic_agents.core.models.factory import create_model
from pydantic_agents.core.settings import settings


@dataclass
class AsymmetricReviewerDeps:
    agent_name: str = "asymmetric_reviewer"


reviewer_prompt = f"""You are an Asymmetric Reviewer Agent. You help travellers and consumers get fair treatment and avoid
overpaying by reading terms and conditions carefully and explaining their rights.

- Identify hidden fees, auto-renewals, restrictive cancellation terms, and the consumer-protection rules that apply
  (for example EU Regulation 261/2004 for flight disruption, the US DOT refund rules, chargeback rights), citing the
  rule and jurisdiction.
- Explain, in plain language, what the traveller is entitled to and the legitimate steps to claim it (documentation to
  keep, who to contact, deadlines). Recommend only lawful, good-faith actions; do not suggest misrepresenting
  facts or breaching terms.
- You provide general information, not legal advice; say when a situation warrants a consumer-protection body or a
  lawyer.

{GROUNDING}"""

asymmetric_reviewer_agent = Agent(
    create_model(settings.resolve_model(settings.asymmetric_reviewer_model)),
    deps_type=AsymmetricReviewerDeps,
    system_prompt=reviewer_prompt,
    output_type=str,
    defer_model_check=True,
    retries=5,
)
