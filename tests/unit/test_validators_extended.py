from pydantic_agents.core.models.core import Artifact
from pydantic_agents.core.validators import CodeValidator


def make_artifact(files: dict) -> Artifact:
    return Artifact(agent_name="test", files=files)


class TestValidateGo:
    async def test_no_go_files_passes(self):
        art = make_artifact({"main.py": "print('hello')"})
        assert await CodeValidator.validate_go_async([art]) is True

    async def test_empty_go_file_skipped(self):
        art = make_artifact({"main.go": ""})
        result = await CodeValidator.validate_go_async([art])
        assert result is False  # go vet fails on empty file


class TestValidateVue:
    async def test_valid_vue_passes(self):
        art = make_artifact({"Comp.vue": "<template><div/></template>\n<script>export default {}</script>"})
        assert await CodeValidator.validate_vue_async([art]) is True

    async def test_missing_template_close_fails(self):
        art = make_artifact({"Comp.vue": "<template><div>\n<script>export default {}</script>"})
        assert await CodeValidator.validate_vue_async([art]) is False

    async def test_missing_script_close_fails(self):
        art = make_artifact({"Comp.vue": "<template><div/></template>\n<script>export default {}"})
        assert await CodeValidator.validate_vue_async([art]) is False

    async def test_tsx_without_export_flagged(self):
        art = make_artifact({"Comp.tsx": "const App = () => {};"})
        assert await CodeValidator.validate_vue_async([art]) is False

    async def test_no_vue_files_passes(self):
        art = make_artifact({"main.py": "x = 1"})
        assert await CodeValidator.validate_vue_async([art]) is True


class TestRunAll:
    async def test_all_valid_passes(self):
        arts = [
            make_artifact({"Comp.vue": "<template><div/></template>\n<script>export default {}</script>"}),
            make_artifact({"main.py": "x = 1"}),
        ]
        assert await CodeValidator.run_all_async(arts) is True

    async def test_any_fails_returns_false(self):
        arts = [
            make_artifact({"Comp.vue": "<template><div/>"}),
        ]
        assert await CodeValidator.run_all_async(arts) is False
