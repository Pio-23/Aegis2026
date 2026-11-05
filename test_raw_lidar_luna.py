import os
import sys
import time

from openai import OpenAI


MODEL = "gpt-5.6-luna"

# Keep headroom for model output and instructions.
MAX_TEST_INPUT_TOKENS = 900_000

# input_text should remain comfortably below API text-size limits.
MAX_TEXT_MB = 8.0


if len(sys.argv) != 2:
    print(
        "Usage:\n"
        "python3 test_raw_lidar_luna.py "
        "/path/to/cloud.txt"
    )
    raise SystemExit(1)


cloud_file = sys.argv[1]


# ----------------------------------------------------------
# CHECK FILE
# ----------------------------------------------------------

if not os.path.exists(cloud_file):

    print(
        f"File not found: {cloud_file}"
    )

    raise SystemExit(1)


file_size_bytes = os.path.getsize(
    cloud_file
)

file_size_mb = (
    file_size_bytes
    / 1024
    / 1024
)


# ----------------------------------------------------------
# READ RAW XYZ TEXT
# ----------------------------------------------------------

print()
print("=" * 60)
print("AEGIS RAW LIDAR TEXT -> LUNA TEST")
print("=" * 60)

print(
    f"File: {cloud_file}"
)

print(
    f"File size: {file_size_mb:.2f} MB"
)


if file_size_mb > MAX_TEXT_MB:

    print()
    print(
        f"STOP: this test sends the XYZ values directly "
        f"as text and the file is larger than "
        f"{MAX_TEXT_MB:.1f} MB."
    )

    print(
        "Create a smaller sampled cloud first."
    )

    raise SystemExit(1)


read_start = time.time()

with open(
    cloud_file,
    "r",
    errors="replace"
) as f:

    cloud_text = f.read()

read_time = (
    time.time()
    - read_start
)


point_count = (
    cloud_text.count("\n")
)

if (
    cloud_text
    and not cloud_text.endswith("\n")
):

    point_count += 1


print(
    f"Point rows: {point_count:,}"
)

print(
    f"Characters: {len(cloud_text):,}"
)

print(
    f"Read time: {read_time:.3f} s"
)

print(
    f"Model: {MODEL}"
)


# ----------------------------------------------------------
# LIDAR INSTRUCTIONS
# ----------------------------------------------------------

lidar_prompt = (
    "You are analyzing a raw AEGIS STL27L LiDAR "
    "Cartesian point cloud.\n\n"

    "Every following line is one directly measured "
    "LiDAR XYZ point in meters:\n\n"

    "X Y Z INTENSITY\n\n"

    "Coordinate system:\n"
    "+X = FRONT\n"
    "-X = BACK\n"
    "+Y = LEFT\n"
    "-Y = RIGHT\n"
    "+Z = UP\n\n"

    "Analyze the raw measured point cloud itself.\n"

    "Do NOT rely on a precomputed directional summary.\n"
    "Do NOT assume Python has already decided which "
    "direction is blocked, clear, or preferable.\n\n"

    "Reason from the XYZ measurements about:\n"
    "- surrounding geometry\n"
    "- approximate measured distances\n"
    "- walls or substantial structures\n"
    "- isolated obstacles\n"
    "- continuous open regions\n"
    "- rover-sized travel paths\n"
    "- which direction appears most useful for "
    "continued exploration\n\n"

    "Do not let a single isolated point determine whether "
    "an entire direction is blocked or open. Consider the "
    "overall distribution of measured points.\n\n"

    "Keep the final analysis concise.\n\n"

    "RAW XYZ POINT CLOUD STARTS BELOW:\n"
    "---------------------------------\n"
)


full_input = (
    lidar_prompt
    + cloud_text
)


client = OpenAI()


# ----------------------------------------------------------
# COUNT PROMPT TOKENS ONLY
# ----------------------------------------------------------

print()
print("Counting instruction tokens...")

prompt_count_start = time.time()

prompt_token_result = (
    client.responses
    .input_tokens
    .count(
        model=MODEL,
        input=lidar_prompt
    )
)

prompt_count_time = (
    time.time()
    - prompt_count_start
)

prompt_tokens = (
    prompt_token_result.input_tokens
)

print(
    f"Instruction tokens: {prompt_tokens:,}"
)


# ----------------------------------------------------------
# COUNT RAW XYZ + PROMPT TOKENS
# ----------------------------------------------------------

print()
print(
    "Counting actual raw XYZ input tokens..."
)

count_start = time.time()

token_result = (
    client.responses
    .input_tokens
    .count(
        model=MODEL,
        input=full_input
    )
)

count_time = (
    time.time()
    - count_start
)

total_input_tokens = (
    token_result.input_tokens
)

estimated_cloud_tokens = (
    total_input_tokens
    - prompt_tokens
)


print()
print("=" * 60)
print("RAW CLOUD SIZE RESULTS")
print("=" * 60)

print(
    f"Total input tokens: "
    f"{total_input_tokens:,}"
)

print(
    f"Instruction tokens: "
    f"{prompt_tokens:,}"
)

print(
    f"Approx. XYZ tokens: "
    f"{estimated_cloud_tokens:,}"
)

print(
    f"Token-count time: "
    f"{count_time:.2f} s"
)


if point_count > 0:

    tokens_per_point = (
        estimated_cloud_tokens
        / point_count
    )

    print(
        f"Approx. tokens / XYZ point: "
        f"{tokens_per_point:.2f}"
    )


# ----------------------------------------------------------
# CHECK WHETHER WE SHOULD RUN INFERENCE
# ----------------------------------------------------------

if (
    total_input_tokens
    > MAX_TEST_INPUT_TOKENS
):

    print()
    print(
        "TEST STOPPED BEFORE LUNA INFERENCE."
    )

    print(
        "The raw XYZ input is too large for "
        "this safe test threshold."
    )

    raise SystemExit(0)


# ----------------------------------------------------------
# ACTUAL LUNA RAW POINT-CLOUD TEST
# ----------------------------------------------------------

print()
print(
    "Sending the actual raw XYZ points "
    "directly to Luna..."
)

inference_start = time.time()


response = (
    client.responses.create(
        model=MODEL,

        reasoning={
            "effort": "none"
        },

        input=full_input,

        max_output_tokens=600
    )
)


inference_time = (
    time.time()
    - inference_start
)


# ----------------------------------------------------------
# RESULTS
# ----------------------------------------------------------

print()
print("=" * 60)
print("LUNA RESULTS")
print("=" * 60)

print(
    f"Model response time: "
    f"{inference_time:.2f} s"
)

print(
    f"Reported input tokens: "
    f"{response.usage.input_tokens:,}"
)

print(
    f"Reported output tokens: "
    f"{response.usage.output_tokens:,}"
)

print()
print("LUNA RAW POINT-CLOUD INTERPRETATION:")
print()
print(response.output_text)
print()

print("=" * 60)