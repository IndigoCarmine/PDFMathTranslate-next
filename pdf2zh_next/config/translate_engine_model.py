import logging
import re
import typing
from dataclasses import dataclass
from inspect import getdoc
from types import NoneType
from typing import Literal
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

# any field in SENSITIVE_FIELDS will be masked in GUI
GUI_SENSITIVE_FIELDS = []
# any field in GUI_PASSWORD_FIELDS will be masked in GUI and treated as password
GUI_PASSWORD_FIELDS = []
# any field in GUI_LOCAL_ONLY_FIELDS is never shown in or accepted from the GUI;
# it can only be set via CLI / env / config file. Used for fields that name a
# local executable, since anyone who can reach the WebUI could otherwise run
# arbitrary commands on the host.
GUI_LOCAL_ONLY_FIELDS = []

logger = logging.getLogger(__name__)


class TranslateEngineSettingError(Exception):
    """Translate engine setting error"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


## Translator configuration classes.
#
# Document content may only leave this machine through the Claude Code CLI.
# Translators that send text to any other service (HTTP APIs, local LLM servers,
# arbitrary external commands) have been removed on purpose; do not add them
# back.
#
# Please note that all translator configurations must be of string type,
# otherwise the GUI will not function properly!
#
# You should implement validation of the translator configuration in validate_settings.
# And complete type conversion (if any) in the corresponding implementation of the translator.


class ClaudeCodeSettings(BaseModel):
    """Claude Code settings"""

    translate_engine_type: Literal["ClaudeCode"] = Field(default="ClaudeCode")
    claude_code_path: str = Field(
        default="claude", description="Path to Claude Code CLI"
    )
    claude_code_model: str = Field(
        default="sonnet", description="Claude Code model to use"
    )

    def validate_settings(self):
        if not self.claude_code_path:
            raise ValueError("Claude Code path is required")
        # Passed to the CLI as a single argument; reject anything that is not a
        # plain model name/alias so it cannot be read as another option.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:\[\]-]*", self.claude_code_model):
            raise ValueError(f"Invalid Claude Code model: {self.claude_code_model}")


GUI_LOCAL_ONLY_FIELDS.append("claude_code_path")


## Please add the translator configuration class above this location.

# 所有翻译引擎
_TRANSLATION_ENGINE_SETTING_TYPES: tuple[type[BaseModel], ...] = (ClaudeCodeSettings,)
TRANSLATION_ENGINE_SETTING_TYPE: TypeAlias = ClaudeCodeSettings

# 不支持的翻译引擎
NOT_SUPPORTED_TRANSLATION_ENGINE_SETTING_TYPE: TypeAlias = NoneType

# 默认翻译引擎
_DEFAULT_TRANSLATION_ENGINE = ClaudeCodeSettings

# The following is magic code,
# if you need to modify it,
# please contact the maintainer!

GUI_SENSITIVE_FIELDS.extend(GUI_PASSWORD_FIELDS)


@dataclass
class TranslationEngineMetadata:
    translate_engine_type: str
    cli_flag_name: str
    cli_detail_field_name: str | None
    setting_model_type: type[BaseModel]
    support_llm: bool

    def __init__(
        self,
        setting_model_type: type[BaseModel],
    ) -> None:
        self.translate_engine_type = setting_model_type.model_fields[
            "translate_engine_type"
        ].default
        self.cli_flag_name = self.translate_engine_type.lower()
        self.cli_detail_field_name = self.cli_flag_name + "_detail"
        self.setting_model_type = setting_model_type
        if len(setting_model_type.model_fields) == 1:
            self.cli_detail_field_name = None
        self.support_llm = (
            (sl := setting_model_type.model_fields.get("support_llm", None))
            and sl.default == "yes"
        ) or False


TRANSLATION_ENGINE_METADATA = [
    TranslationEngineMetadata(
        setting_model_type=arg,
    )
    for arg in _TRANSLATION_ENGINE_SETTING_TYPES
]

TRANSLATION_ENGINE_METADATA_MAP = {
    metadata.translate_engine_type: metadata for metadata in TRANSLATION_ENGINE_METADATA
}


# auto check duplicate translation engine metadata
assert len(TRANSLATION_ENGINE_METADATA_MAP) == len(TRANSLATION_ENGINE_METADATA), (
    "Duplicate translation engine metadata"
)

# auto check duplicate cli flag name and cli detail field name
dedup_set = set()
for metadata in TRANSLATION_ENGINE_METADATA:
    if metadata.cli_flag_name in dedup_set:
        raise ValueError(f"Duplicate cli flag name: {metadata.cli_flag_name}")
    dedup_set.add(metadata.cli_flag_name)
    if metadata.cli_detail_field_name and metadata.cli_detail_field_name in dedup_set:
        raise ValueError(
            f"Duplicate cli detail field name: {metadata.cli_detail_field_name}"
        )
    dedup_set.add(metadata.cli_detail_field_name)
del dedup_set


# Term extraction can only use LLM-capable translation engines. Claude Code is
# not one, so there are currently no separate term extraction engines and term
# extraction never sends text anywhere on its own.
_TERM_EXTRACTION_ENGINE_SETTING_TYPE: typing.Any = NoneType
for metadata in TRANSLATION_ENGINE_METADATA:
    if not metadata.support_llm:
        continue
    if _TERM_EXTRACTION_ENGINE_SETTING_TYPE is NoneType:
        _TERM_EXTRACTION_ENGINE_SETTING_TYPE = metadata.setting_model_type
    else:
        _TERM_EXTRACTION_ENGINE_SETTING_TYPE = (
            _TERM_EXTRACTION_ENGINE_SETTING_TYPE | metadata.setting_model_type
        )

# 术语提取引擎：仅包含 support_llm == \"yes\" 的翻译引擎设置类型
TERM_EXTRACTION_ENGINE_SETTING_TYPE: TypeAlias = _TERM_EXTRACTION_ENGINE_SETTING_TYPE


def _build_term_setting_model(
    setting_model_type: type[BaseModel],
) -> type[BaseModel]:
    """Dynamically build a term-extraction settings model with prefixed fields."""
    fields: dict[str, tuple[typing.Any, Field]] = {}
    base_to_term_field_map: dict[str, str] = {}

    for name, model_field in setting_model_type.model_fields.items():
        # Keep discriminator-related fields unchanged
        if name in ("translate_engine_type", "support_llm"):
            new_name = name
        else:
            new_name = f"term_{name}"

        base_to_term_field_map[name] = new_name

        fields[new_name] = (
            model_field.annotation,
            Field(
                default=model_field.default,
                description=model_field.description,
                default_factory=model_field.default_factory,
                alias=model_field.alias,
                discriminator=model_field.discriminator,
            ),
        )

    term_model_name = f"Term{setting_model_type.__name__}"
    TermModel = create_model(term_model_name, **fields)  # type: ignore[arg-type]  # noqa: N806

    # Set a meaningful docstring for the dynamically created term settings model
    # so that inspect.getdoc(TermModel) returns helpful information in CLI help.
    base_doc = getdoc(setting_model_type) or setting_model_type.__doc__ or ""
    if base_doc:
        TermModel.__doc__ = f"Term settings based on: {base_doc}"
    else:
        TermModel.__doc__ = (
            "Term settings model based on the base engine settings model."
        )

    def to_base_settings(self) -> BaseModel:
        """Convert term settings back to the base engine settings model."""
        data: dict[str, typing.Any] = {}
        for base_name, term_name in base_to_term_field_map.items():
            data[base_name] = getattr(self, term_name)
        return setting_model_type(**data)

    TermModel.to_base_settings = to_base_settings  # type: ignore[attr-defined]
    return TermModel


@dataclass
class TermTranslationEngineMetadata:
    translate_engine_type: str
    cli_flag_name: str
    cli_detail_field_name: str | None
    term_setting_model_type: type[BaseModel]


TERM_EXTRACTION_ENGINE_METADATA: list[TermTranslationEngineMetadata] = []

for metadata in TRANSLATION_ENGINE_METADATA:
    if not metadata.support_llm:
        continue
    term_setting_model_type = _build_term_setting_model(metadata.setting_model_type)
    TERM_EXTRACTION_ENGINE_METADATA.append(
        TermTranslationEngineMetadata(
            translate_engine_type=metadata.translate_engine_type,
            cli_flag_name=metadata.cli_flag_name,
            cli_detail_field_name=metadata.cli_detail_field_name,
            term_setting_model_type=term_setting_model_type,
        )
    )

TERM_EXTRACTION_ENGINE_METADATA_MAP = {
    metadata.translate_engine_type: metadata
    for metadata in TERM_EXTRACTION_ENGINE_METADATA
}


DEFAULT_TRANSLATION_ENGINE_METADATA = TRANSLATION_ENGINE_METADATA_MAP[
    _DEFAULT_TRANSLATION_ENGINE.model_fields["translate_engine_type"].default
]

if __name__ == "__main__":
    print(TRANSLATION_ENGINE_METADATA_MAP)
