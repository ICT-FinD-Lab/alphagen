def _save_survival_data(alpha_survival_history, save_path, timestamp, test_ic, verbose=0):
    """Save alpha survival tracking data and generate plots."""
    survival_filepath = os.path.join(save_path, f"{timestamp}_alpha_survival.json")
    print(f"[DEBUG] _save_survival_data called, saving to {survival_filepath}")

    # Prepare survival data
    survival_data = {
        'metadata': {
            'total_alphas_created': len(alpha_survival_history),
            'llm_alphas': sum(1 for d in alpha_survival_history.values() if d['origin'] == 'llm'),
            'rl_alphas': sum(1 for d in alpha_survival_history.values() if d['origin'] == 'rl'),
            'total_training_steps': test_ic
        },
        'alphas': list(alpha_survival_history.values())
    }

    with open(survival_filepath, 'w') as f:
        json.dump(survival_data, f, indent=2)

    if verbose > 0:
        print(f'Saved alpha survival data to {survival_filepath}')

    # Generate visualization
    _plot_alpha_survival(alpha_survival_history, save_path, timestamp, test_ic, verbose=verbose)

def _plot_alpha_survival(alpha_survival_history, save_path, timestamp, test_ic, verbose=0):
    """Generate a plot of alpha survival timelines."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        if verbose > 0:
            print('Matplotlib not available, skipping survival plot generation')
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort all alphas by birth timestep (when they entered the pool)
    sorted_alphas = sorted(alpha_survival_history.values(), key=lambda d: d['birth_timestep'])

    # Plot alphas in order of birth timestep
    color_map = {'llm': 'coral', 'rl': 'steelblue', 'improvement': 'lightgreen'}
    for idx, alpha in enumerate(sorted_alphas):
        birth = alpha['birth_timestep']
        death = alpha['death_timestep'] if alpha['death_timestep'] is not None else birth
        origin = alpha['origin']
        color = color_map.get(origin, 'gray')
        ax.barh(idx, death - birth, left=birth, height=0.8, color=color, alpha=0.7)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Alpha Index ', fontsize=12)
    ax.set_title('Alpha Survival Timeline', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    llm_alphas = [d for d in sorted_alphas if d['origin'] == 'llm']
    rl_alphas = [d for d in sorted_alphas if d['origin'] == 'rl']
    improvement_alphas = [d for d in sorted_alphas if d['origin'] == 'improvement']
    
    rl_patch = mpatches.Patch(color='steelblue', alpha=0.7, label=f'RL Alphas (n={len(rl_alphas)})')
    llm_patch = mpatches.Patch(color='coral', alpha=0.7, label=f'LLM Alphas (n={len(llm_alphas)})')
    improvement_patch = mpatches.Patch(color='lightgreen', alpha=0.7, label=f'Improvement Alphas (n={len(improvement_alphas)})')
    ax.legend(handles=[rl_patch, llm_patch, improvement_patch], loc='upper right')

    plot_filepath = os.path.join(save_path, f"{timestamp}_alpha_survival.png")
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose > 0:
        print(f'Saved survival plot to {plot_filepath}')
import os
import json
import numpy as np

def init_llm_alpha_stats():
    return {
        'original': {'valid': 0, 'invalid': 0, 'fixed': 0, 'complexity': [], 'ic': [], 'ic_after': [], 'ic_diff': []},
        'improvement': {'valid': 0, 'invalid': 0, 'fixed': 0, 'complexity': [], 'ic': [], 'ic_after': [], 'ic_diff': []},
        'survival': {'rl': [], 'llm': [], 'improvement': []}
    }

def record_llm_alpha(llm_alpha_stats, kind, valid, fixed, expr, ic_before=None, ic_after=None):
    # kind: 'original' or 'improvement'
    if valid:
        llm_alpha_stats[kind]['valid'] += 1
    else:
        llm_alpha_stats[kind]['invalid'] += 1
    if fixed:
        llm_alpha_stats[kind]['fixed'] += 1
    # Complexity: use expr complexity() if available, else len(str(expr))
    try:
        comp = expr.complexity() if hasattr(expr, 'complexity') else len(str(expr))
    except Exception:
        comp = len(str(expr))
    llm_alpha_stats[kind]['complexity'].append(comp)
    if ic_before is not None:
        llm_alpha_stats[kind]['ic'].append(ic_before)
    if ic_after is not None:
        llm_alpha_stats[kind]['ic_after'].append(ic_after)
    if ic_before is not None and ic_after is not None:
        llm_alpha_stats[kind]['ic_diff'].append(ic_after - ic_before)

def record_survival(llm_alpha_stats, alpha_survival_history):
    # Collect survival steps for each origin
    for d in alpha_survival_history.values():
        origin = d.get('origin', 'rl')
        surv = d.get('survival_steps', 0)
        if origin == 'llm':
            llm_alpha_stats['survival']['llm'].append(surv)
        elif origin == 'improvement':
            llm_alpha_stats['survival']['improvement'].append(surv)
        else:
            llm_alpha_stats['survival']['rl'].append(surv)

def save_llm_alpha_stats(llm_alpha_stats, save_dir, verbose=0):
    stats = {}
    for kind in ['original', 'improvement']:
        c = np.array(llm_alpha_stats[kind]['complexity'])
        ic = np.array(llm_alpha_stats[kind]['ic'])
        ic_after = np.array(llm_alpha_stats[kind]['ic_after'])
        ic_diff = np.array(llm_alpha_stats[kind]['ic_diff'])
        stats[kind] = {
            'valid': llm_alpha_stats[kind]['valid'],
            'invalid': llm_alpha_stats[kind]['invalid'],
            'fixed': llm_alpha_stats[kind]['fixed'],
            'valid_pct': float(llm_alpha_stats[kind]['valid']) / max(1, llm_alpha_stats[kind]['valid'] + llm_alpha_stats[kind]['invalid']),
            'complexity': {
                'mean': float(np.mean(c)) if c.size else None,
                'std': float(np.std(c)) if c.size else None,
                'min': float(np.min(c)) if c.size else None,
                'max': float(np.max(c)) if c.size else None
            },
            'ic': {
                'mean': float(np.mean(ic)) if ic.size else None,
                'std': float(np.std(ic)) if ic.size else None,
                'min': float(np.min(ic)) if ic.size else None,
                'max': float(np.max(ic)) if ic.size else None
            },
            'ic_after': {
                'mean': float(np.mean(ic_after)) if ic_after.size else None,
                'std': float(np.std(ic_after)) if ic_after.size else None,
                'min': float(np.min(ic_after)) if ic_after.size else None,
                'max': float(np.max(ic_after)) if ic_after.size else None
            },
            'ic_diff': {
                'mean': float(np.mean(ic_diff)) if ic_diff.size else None,
                'std': float(np.std(ic_diff)) if ic_diff.size else None,
                'min': float(np.min(ic_diff)) if ic_diff.size else None,
                'max': float(np.max(ic_diff)) if ic_diff.size else None
            }
        }
    # Survival stats
    for k in ['rl', 'llm', 'improvement']:
        arr = np.array(llm_alpha_stats['survival'][k])
        stats[f'survival_{k}'] = {
            'mean': float(np.mean(arr)) if arr.size else None,
            'std': float(np.std(arr)) if arr.size else None,
            'min': float(np.min(arr)) if arr.size else None,
            'max': float(np.max(arr)) if arr.size else None,
            'count': int(arr.size)
        }
    # Save
    out_path = os.path.join(save_dir, 'llm_alpha_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    if verbose > 0:
        print(f'Saved LLM alpha stats to {out_path}')
