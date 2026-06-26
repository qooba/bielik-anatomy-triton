"""
Interactive chat with Bielik model using Triton kernels

This chat script uses the new Bielik implementation with ~100% Triton kernels.
Note: Without KV cache, generation recomputes full attention each token (slower but correct).
KV cache will be added in a future episode for faster generation.
"""

import torch
from transformers import AutoTokenizer
import sys
import os
import time

# Add parent directory to path to import bielik module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bielik import BielikModel


class BielikChat:
    def __init__(self, model_id="speakleash/Bielik-1.5B-v3.0-Instruct", device='cuda', dtype=torch.bfloat16):
        """Initialize chat with Bielik model"""
        print("=" * 80)
        print("BIELIK INTERACTIVE CHAT")
        print("Using Triton Kernels (~100% GPU compute)")
        print("=" * 80)

        print(f"\nLoading model: {model_id}")
        self.model = BielikModel.from_pretrained(model_id, device=device, dtype=dtype)
        print(f"Model loaded: {self.model.config.num_layers} layers, "
              f"{self.model.config.hidden_size}D hidden size")

        print(f"\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Tokenizer loaded: {self.tokenizer.vocab_size:,} tokens")

        self.device = device
        self.conversation_history = []

        # Check if tokenizer has chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            print(f"Using tokenizer's chat template")
            self.use_chat_template = True
        else:
            print(f"No chat template found, using simple format")
            self.use_chat_template = False


    def format_prompt(self, user_message):
        """Format prompt with chat history"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        if self.use_chat_template:
            # Use tokenizer's chat template
            prompt = self.tokenizer.apply_chat_template(
                self.conversation_history,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Simple fallback format
            prompt = ""
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    prompt += f"Użytkownik: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Asystent: {msg['content']}\n"
            prompt += "Asystent:"

        return prompt

    def generate_response_stream(self, prompt, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.95):
        """Generate response from model, yielding tokens one by one

        Note: Without KV cache, this recomputes full attention for each token.
        This is O(N^2) in sequence length but ensures correctness.
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_len = input_ids.shape[1]

        # Generate
        with torch.no_grad():
            generated_ids = input_ids[0].tolist()

            for _ in range(max_new_tokens):
                # Get logits (recomputes full attention - no KV cache yet)
                current_input = torch.tensor([generated_ids], device=self.device)
                logits = self.model.forward(current_input)
                next_token_logits = logits[0, -1, :]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    # Greedy
                    next_token = next_token_logits.argmax().item()

                # Check for EOS
                eos_ids = self.tokenizer.eos_token_id if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id]
                if next_token in eos_ids:
                    break

                generated_ids.append(next_token)

                # Decode and yield the new token
                # We decode the entire sequence to handle multi-byte tokens correctly
                full_text = self.tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=True)
                yield full_text

    def chat(self, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.95, show_speed=True):
        """Interactive chat loop"""
        print("\n" + "=" * 80)
        print("CHAT MODE")
        print("=" * 80)
        print("\nInstrukcje:")
        print("  - Wpisz swoją wiadomość i naciśnij Enter")
        print("  - Wpisz 'exit' lub 'quit' aby zakończyć")
        print("  - Wpisz 'clear' aby wyczyścić historię rozmowy")
        print("  - Wpisz 'history' aby zobaczyć historię rozmowy")
        print(f"\nParametry generacji:")
        print(f"  - Max tokens: {max_new_tokens}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Top-k: {top_k}")
        print(f"  - Top-p: {top_p}")
        print(f"  - Show speed: {show_speed}")
        print("\n" + "=" * 80)

        while True:
            try:
                # Get user input
                user_input = input("\n👤 Ty: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Do widzenia!")
                    break

                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("✓ Historia rozmowy wyczyszczona")
                    continue

                if user_input.lower() == 'history':
                    print("\n📜 Historia rozmowy:")
                    for msg in self.conversation_history:
                        role = "Ty" if msg["role"] == "user" else "Bielik"
                        print(f"  {role}: {msg['content']}")
                    continue

                # Format prompt with history
                prompt = self.format_prompt(user_input)

                # Generate response with streaming
                print("\n🤖 Bielik: ", end="", flush=True)

                response = ""
                prev_text = ""
                token_count = 0
                start_time = time.time()

                for current_text in self.generate_response_stream(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                ):
                    # Print only the new characters
                    new_chars = current_text[len(prev_text):]
                    print(new_chars, end="", flush=True)
                    prev_text = current_text
                    response = current_text
                    token_count += 1

                end_time = time.time()
                elapsed_time = end_time - start_time

                # Print newline after response is complete
                print()

                # Show speed metrics if enabled
                if show_speed and token_count > 0:
                    tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0
                    ms_per_token = (elapsed_time * 1000) / token_count if token_count > 0 else 0
                    print(f"\n⚡ Speed: {tokens_per_sec:.2f} tokens/sec | {ms_per_token:.2f} ms/token | {token_count} tokens in {elapsed_time:.2f}s")

                # Clean up response
                response = response.strip()

                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

            except KeyboardInterrupt:
                print("\n\n👋 Przerwano przez użytkownika. Do widzenia!")
                break
            except Exception as e:
                print(f"\n❌ Błąd: {e}")
                import traceback
                traceback.print_exc()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Interactive chat with Bielik model (Triton kernels)')
    parser.add_argument(
        '--model',
        type=str,
        default='speakleash/Bielik-1.5B-v3.0-Instruct',
        help='Model ID or path'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (0=greedy, higher=more random)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--no-speed',
        action='store_true',
        help='Disable speed metrics display'
    )

    args = parser.parse_args()

    # Create chat instance
    chat = BielikChat(
        model_id=args.model,
        device=args.device,
        dtype=torch.bfloat16
    )

    # Start chat
    chat.chat(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        show_speed=not args.no_speed
    )


if __name__ == '__main__':
    main()
