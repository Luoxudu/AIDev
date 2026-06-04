"""系统提示词 CRUD。"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.agent.memory import load_raw_profile, save_raw_profile

router = APIRouter()


class SystemPromptUpdate(BaseModel):
    content: str


@router.get("/system-prompt")
def get_system_prompt():
    profile = load_raw_profile()
    return {"content": profile.get("system_prompt", "")}


@router.put("/system-prompt")
def update_system_prompt(body: SystemPromptUpdate):
    profile = load_raw_profile()
    profile["system_prompt"] = body.content
    save_raw_profile(profile)

    from src.agent.agent import reset_web_agent
    reset_web_agent()

    return {"ok": True}
