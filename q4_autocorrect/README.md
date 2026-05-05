# Question 4: Probabilistic AutoCorrect

This project implements a spelling correction model using Shakespeare's text as
its vocabulary and frequency source.

## Files

- `submit.py`: contains the required functions and the `autoCorrect` model.
- `shakespeare.txt`: expected corpus file used to build vocabulary and frequencies.
- `requirements.txt`: minimal dependencies.

## Instructions

1. Place `shakespeare.txt` in the `q4_autocorrect/` directory.
2. Run the module with:

```bash
cd q4_autocorrect
python submit.py
```

## Notes

- `delete_letter`, `replace_letter`, and `insert_letter` generate all one-edit candidates.
- `edit_one_letter` returns the union of these candidates.
- `edit_two_letters` returns all two-edit-away candidates.
- `autoCorrect` chooses the valid correction with the lowest edit cost and highest corpus probability.
