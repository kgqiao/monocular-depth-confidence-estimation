# Fair Matching

This script assigns the five team members to the three project roles from their ranked preferences. Each person ranks every role. The script then checks every possible assignment and picks the fairest one with a rule that anyone can verify.

## The rule

The script compares assignments in this order:

1. Fewest people with their last choice.
2. Then lowest total rank (1st = 1, 2nd = 2, 3rd = 3).
3. Then a random pick among the assignments that are still tied.

Step 1 protects the person who is worst off. If no assignment avoids last choices, the script still gives as few people their last choice as it can. Ranking by total alone does not do this: it can give two people their last choice when one is enough. Step 2 then gives the team as many top choices as possible.

The script prints every tied assignment and the random seed, so the team can see all the equal options and repeat the pick.

## Usage

1. In `preferences.json`, write each member's ranking as a list of role names, top choice first:

   ```json
   "Nathan A": ["Evaluation", "Datasets and Training", "Cross team coordinator"]
   ```

2. Run the script:

   ```sh
   python3 match.py
   ```

3. To repeat a random pick, use the seed that the script printed:

   ```sh
   python3 match.py --seed 123456
   ```

The script needs only Python 3 and works for any number of people and role sizes. To use a different file, give its path: `python3 match.py other.json`.

## Limitation

Any rank-based method can be gamed. For example, a member can rank a popular role last to reduce competition for it. The method works only if each member reports their true preferences.
