    # Define the name mapping dictionary from the notebook
name_fix_dict = {
        # Direct Matches
        'Pushups': 'pushups',
        'Jumping Jacks': 'jumpingjacks',
        'Sit-ups': 'situps',
        'Bicep Curl': 'bicepcurls',
        'Lateral Raise': 'lateralshoulderraises',

        # Partial Matches (Closest Equivalent)
        'Squat': 'squats',
        'Squat Jump': 'squats',
        'Wall Squat': 'squats',
        'Dumbbell Squat (hands at side)': 'squats',
        'Squat (arms in front of body, parallel to ground)': 'squats',
        'Squat (hands behind head)': 'squats',
        'Squat (kettlebell / goblet)': 'squats',

        'Pushup (knee or foot variation)': 'pushups',

        'Shoulder Press (dumbbell)': 'dumbbellshoulderpress',
        'Squat Rack Shoulder Press': 'dumbbellshoulderpress',

        'Lunge (alternating both legs, weight optional)': 'lunges',
        'Walking lunge': 'lunges',

        'Dumbbell Row (knee on bench) (label spans both arms)': 'dumbbellrows',
        'Dumbbell Row (knee on bench) (left arm)': 'dumbbellrows',
        'Dumbbell Row (knee on bench) (right arm)': 'dumbbellrows',
        'Dumbbell Deadlift Row': 'dumbbellrows',

        'Overhead Triceps Extension': 'tricepextensions',
        'Triceps extension (lying down)': 'tricepextensions',
        'Triceps Kickback (knee on bench) (label spans both arms)': 'tricepextensions',
        'Triceps Kickback (knee on bench) (left arm)': 'tricepextensions',
        'Triceps Kickback (knee on bench) (right arm)': 'tricepextensions',
        'Triceps extension (lying down) (left arm)': 'tricepextensions',
        'Triceps extension (lying down) (right arm)': 'tricepextensions',

        # No Direct Match
        'Non-Exercise': 'rest',
        'Device on Table': 'rest',
        'Tap Left Device': 'rest',
        'Tap Right Device': 'rest',
        'Arm Band Adjustment': 'rest',
        'Initial Activity': 'rest',
        'Invalid': 'rest',
        'Note': 'rest',
        'Unlisted Exercise': 'rest',
        'non-e': 'rest',
        'nonexercise': 'rest',
        'staticstretch(atyourownpace)': 'staticstretch',
        'two-armdumbbellcurl(botharms,notalternating)': 'bicepcurls',
        'wallballs': 'wallball',
        'dumbbell_shoulder_press': 'dumbbellshoulderpress',
        'lateral_shoulder_raises': 'lateralshoulderraises',
        'sit-up(handspositionedbehindhead)': 'situps',
        'null': 'rest'
    }
    
    # Column rename mapping from the notebook
rename_dict = {
        'label': 'activity_name',
        'gyrX': 'gyr_X',
        'gyrY': 'gyr_Y', 
        'gyrZ': 'gyr_Z',
        'accX': 'acc_X', 
        'accY': 'acc_Y', 
        'accZ': 'acc_Z',
        'predicted_exercise': 'activity_name', 
        'actual_exercise': 'actual_activity_name',
        'actual_reps': 'repetitions',
    }