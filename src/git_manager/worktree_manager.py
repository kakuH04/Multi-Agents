"""Git worktree management for parallel task execution"""
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Dict
import logging
import git

logger = logging.getLogger(__name__)


class WorktreeManager:
    """Manage git worktrees for parallel task execution"""

    def __init__(self, base_repo_path: str, worktree_base: str = "./worktrees"):
        self.base_repo_path = Path(base_repo_path)
        self.worktree_base = Path(worktree_base)
        self.worktree_base.mkdir(parents=True, exist_ok=True)

        # Verify base repo is a git repository
        try:
            self.repo = git.Repo(self.base_repo_path)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"{base_repo_path} is not a git repository")

    def create_worktree(self, name: str, branch: Optional[str] = None) -> Path:
        """
        Create a new git worktree

        Args:
            name: Name for the worktree
            branch: Branch to checkout (creates new branch if doesn't exist)

        Returns:
            Path to the created worktree
        """
        worktree_path = self.worktree_base / name

        if worktree_path.exists():
            logger.warning(f"Worktree {name} already exists at {worktree_path}")
            return worktree_path

        # Create branch name if not provided
        if branch is None:
            branch = f"task/{name}"

        try:
            # Check if branch exists
            branch_exists = branch in [b.name for b in self.repo.branches]

            if branch_exists:
                # Add worktree with existing branch
                cmd = ["git", "worktree", "add", str(worktree_path), branch]
            else:
                # Add worktree with new branch
                cmd = ["git", "worktree", "add", "-b", branch, str(worktree_path)]

            subprocess.run(
                cmd,
                cwd=self.base_repo_path,
                check=True,
                capture_output=True,
                text=True
            )

            logger.info(f"Created worktree '{name}' at {worktree_path} on branch '{branch}'")
            return worktree_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create worktree: {e.stderr}")
            raise

    def remove_worktree(self, name: str, force: bool = False):
        """
        Remove a git worktree

        Args:
            name: Name of the worktree to remove
            force: Force removal even with uncommitted changes
        """
        worktree_path = self.worktree_base / name

        if not worktree_path.exists():
            logger.warning(f"Worktree {name} does not exist")
            return

        try:
            cmd = ["git", "worktree", "remove", str(worktree_path)]
            if force:
                cmd.append("--force")

            subprocess.run(
                cmd,
                cwd=self.base_repo_path,
                check=True,
                capture_output=True,
                text=True
            )

            logger.info(f"Removed worktree '{name}'")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove worktree: {e.stderr}")
            # Try manual cleanup if force was requested
            if force and worktree_path.exists():
                shutil.rmtree(worktree_path)
                # Prune worktree from git
                subprocess.run(
                    ["git", "worktree", "prune"],
                    cwd=self.base_repo_path,
                    check=True
                )
                logger.info(f"Force removed worktree '{name}' manually")

    def list_worktrees(self) -> List[Dict[str, str]]:
        """
        List all worktrees

        Returns:
            List of worktree information dictionaries
        """
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=self.base_repo_path,
                check=True,
                capture_output=True,
                text=True
            )

            worktrees = []
            current_worktree = {}

            for line in result.stdout.strip().split('\n'):
                if not line:
                    if current_worktree:
                        worktrees.append(current_worktree)
                        current_worktree = {}
                    continue

                if line.startswith('worktree '):
                    current_worktree['path'] = line.split(' ', 1)[1]
                elif line.startswith('HEAD '):
                    current_worktree['HEAD'] = line.split(' ', 1)[1]
                elif line.startswith('branch '):
                    current_worktree['branch'] = line.split(' ', 1)[1]
                elif line.startswith('bare'):
                    current_worktree['bare'] = True

            if current_worktree:
                worktrees.append(current_worktree)

            return worktrees

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list worktrees: {e.stderr}")
            return []

    def cleanup_worktrees(self, keep_names: List[str] = None):
        """
        Clean up worktrees, optionally keeping specific ones

        Args:
            keep_names: List of worktree names to keep
        """
        keep_names = keep_names or []
        worktrees = self.list_worktrees()

        for wt in worktrees:
            path = Path(wt['path'])
            # Skip base repository
            if path == self.base_repo_path:
                continue

            # Check if this worktree should be kept
            name = path.name
            if name in keep_names:
                logger.info(f"Keeping worktree: {name}")
                continue

            # Remove worktree
            self.remove_worktree(name, force=True)

    def commit_and_push(self, worktree_name: str, message: str, push: bool = True) -> bool:
        """
        Commit changes in a worktree and optionally push

        Args:
            worktree_name: Name of the worktree
            message: Commit message
            push: Whether to push changes

        Returns:
            True if successful, False otherwise
        """
        worktree_path = self.worktree_base / worktree_name

        if not worktree_path.exists():
            logger.error(f"Worktree {worktree_name} does not exist")
            return False

        try:
            # Open worktree as a repo
            wt_repo = git.Repo(worktree_path)

            # Add all changes
            wt_repo.git.add(A=True)

            # Check if there are changes to commit
            if not wt_repo.is_dirty() and not wt_repo.untracked_files:
                logger.info(f"No changes to commit in worktree {worktree_name}")
                return True

            # Commit
            wt_repo.index.commit(message)
            logger.info(f"Committed changes in worktree {worktree_name}")

            # Push if requested
            if push:
                origin = wt_repo.remote('origin')
                current_branch = wt_repo.active_branch.name
                origin.push(current_branch)
                logger.info(f"Pushed branch {current_branch} to origin")

            return True

        except Exception as e:
            logger.error(f"Failed to commit/push in worktree {worktree_name}: {e}")
            return False

    def get_worktree_path(self, name: str) -> Optional[Path]:
        """Get the path to a worktree by name"""
        worktree_path = self.worktree_base / name
        return worktree_path if worktree_path.exists() else None
