# Read the file, keep lines 1-183, rewrite the rest cleanly
with open("plot_pareto.py", "r") as f:
    lines = f.readlines()

tail = """
    xl = ax.get_xlim()[0]
    xr = ax.get_xlim()[1]
    yl = ax.get_ylim()[0]
    yr = ax.get_ylim()[1]
    ax.annotate("", xy=(xl, yl), xytext=(xl + (xr-xl)*0.06, yl),
                arrowprops=dict(arrowstyle="<-", color="black", lw=0.8))
    ax.annotate("", xy=(xl, yl), xytext=(xl, yl + (yr-yl)*0.06),
                arrowprops=dict(arrowstyle="<-", color="black", lw=0.8))

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
"""

with open("plot_pareto.py", "w") as f:
    f.writelines(lines[:183])
    f.write(tail)

print("Done")
