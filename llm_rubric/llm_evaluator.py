from jinja2 import Template


def make_message(
    instruction_prompt: str,
    metaprompt_template: str,
    evaluation_text: str,
    criterion: str,
) -> list[dict[str,str]]:
    template = Template(metaprompt_template)

    return [
        {
            "role": "system",
            "content": instruction_prompt,
        },
        {
            "role": "user",
            "content": template.render(
                evaluation_text=evaluation_text,
                criterion=criterion,
            )
        }

    ] 


def get_llm_api_output(
    messages,
):
    expected_tokens = ["1", "2", "3", "4"]
    model = "gpt-3.5-turbo-16k"
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        top_logprobs=5,
        logprobs=True,
    )
