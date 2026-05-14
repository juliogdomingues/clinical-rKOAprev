
import pandas as pd
import oarsi_data as p

def main():
    print("Loading data...")
    # Load data using the project's standard loader
    df = p.load_and_prep_data("./base_stata/stataToCsvMG.csv")
    
    # --- Prevalence by Knee ---
    n_knees = len(df)
    n_koa_knees = df['oa_knee'].sum()
    prev_knee = (n_koa_knees / n_knees) * 100
    
    print(f"\nTotal Knees: {n_knees}")
    print(f"KOA Knees: {int(n_koa_knees)}")
    print(f"Prevalence by Knee: {prev_knee:.2f}%")
    
    # --- Prevalence by Participant ---
    # Group by participant ID
    # A participant has KOA if max(oa_knee) == 1 (meaning at least one knee has it)
    participant_max = df.groupby('idelsa')['oa_knee'].max()
    
    n_participants = len(participant_max)
    n_koa_participants = participant_max.sum()
    prev_participant = (n_koa_participants / n_participants) * 100
    
    print(f"\nTotal Participants: {n_participants}")
    print(f"Participants with KOA (>=1 knee): {int(n_koa_participants)}")
    print(f"Prevalence by Participant: {prev_participant:.2f}%")

if __name__ == "__main__":
    main()
