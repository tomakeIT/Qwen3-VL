from typing import List, Dict, Any, Tuple, Optional

def build_prompt_with_reference_multiview(
    ref_img_paths: List[str],
    ref_progress_ints: List[int],
    target_img_paths_t1: List[str],
    target_img_paths_t2: List[str],
    reference_view_names: List[str],
    target_view_names: List[str],
    task_desc: str,
) -> Tuple[List[str], str]:
    """构造一条 Qwen 风格样本"""
    human_prompt: List[str] = []
    human_prompt.append(
        "You are a robotic task progress evaluator.\n\n"
        f"Task description: {task_desc}\n\n"
        "You will first see several time steps from a reference demonstration of this task.\n"
        "Each time step contains multiple synchronized camera views and is annotated with its\n"
        "absolute task completion percentage (an integer between 0 and 100).\n"
        "Then you will see two time steps (Image-1 and Image-2) from another episode of the SAME task,\n"
        "also with multiple camera views.\n\n"
    )

    if ref_img_paths and ref_progress_ints:
        human_prompt.append(
            "Here are example time steps from a reference demonstration with their absolute completion percentages.\n"
            "For each time step, the views are given in the following fixed order:\n"
            "  - " + ", ".join(reference_view_names) + "\n\n"
        )
        num_ref_views = len(reference_view_names)
        num_steps = len(ref_progress_ints)
        assert len(ref_img_paths) == num_ref_views * num_steps
        for step_idx in range(num_steps):
            prog = ref_progress_ints[step_idx]
            human_prompt.append(f"Reference Time Step {step_idx + 1}:\n")
            for v in reference_view_names:
                human_prompt.append(f"- View {v}: <image>\n")
            human_prompt.append(f"The task completion percentage for this time step is {prog:d}%.\n\n")
    else:
        human_prompt.append("No reference demonstration is available for this task. Please rely on your general understanding.\n\n")

    human_prompt.append(
        "Now consider another episode of the SAME task.\n"
        "We will show you two time steps from this episode, each with the following camera views\n"
        "in the exact order they are provided:\n"
        "  - " + ", ".join(target_view_names) + "\n\n"
    )
    human_prompt.append("Image-1 (earlier or reference time step):\n")
    for v in target_view_names:
        human_prompt.append(f"- View {v}: <image>\n")
    human_prompt.append("\nImage-2 (another time step of the same episode):\n")
    for v in target_view_names:
        human_prompt.append(f"- View {v}: <image>\n")
    human_prompt.append(
        "\nLet the task completion percentages of Image-1 and Image-2 be P1 and P2 (both between 0 and 100).\n"
        "Your job is to output the integer delta progress D = round(P2 - P1),\n"
        "which must be an integer between -100 and 100 (inclusive).\n\n"
        "OUTPUT REQUIREMENT:\n"
        "- Return ONLY the integer D (with optional leading '+' or '-' sign),\n"
        "  e.g., +5, -13, 0, +42, -100, +100.\n"
        "- Do NOT output any explanation, percent sign, or extra text.\n"
    )

    human_str: str = "".join(human_prompt)
    img_paths = ref_img_paths + target_img_paths_t1 + target_img_paths_t2

    return img_paths, human_str

