#!/usr/bin/env pwsh
<#
.SYNOPSIS
    VIVTransformer GitHub Pages Setup Tool

.DESCRIPTION
    Automated setup tool for VIVTransformer GitHub Pages documentation site

.PARAMETER Mode
    Setup mode: auto, interactive, install, preview, status, menu

.EXAMPLE
    .\setup.ps1
    Interactive menu mode

.EXAMPLE
    .\setup.ps1 -Mode auto
    Automatic setup mode
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("auto", "interactive", "install", "preview", "status", "menu")]
    [string]$Mode = "menu",
    
    [Parameter(Mandatory=$false)]
    [string]$Username,
    
    [Parameter(Mandatory=$false)]
    [string]$Repository,
    
    [Parameter(Mandatory=$false)]
    [string]$Title,
    
    [Parameter(Mandatory=$false)]
    [string]$Description,
    
    [Parameter(Mandatory=$false)]
    [string]$AuthorName,
    
    [Parameter(Mandatory=$false)]
    [string]$AuthorEmail,
    
    [Parameter(Mandatory=$false)]
    [string]$Branch = "main"
)

# Set console encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Helper functions
function Write-ColorText {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

# Check if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Run command safely
function Invoke-SafeCommand {
    param(
        [string]$Command,
        [string]$WorkingDirectory = $PWD,
        [bool]$CaptureOutput = $true
    )
    
    try {
        if ($CaptureOutput) {
            $result = Invoke-Expression $Command 2>&1
            return @{
                Success = $LASTEXITCODE -eq 0
                Output = $result -join "`n"
                ExitCode = $LASTEXITCODE
            }
        } else {
            Push-Location $WorkingDirectory
            Invoke-Expression $Command
            $exitCode = $LASTEXITCODE
            Pop-Location
            return @{
                Success = $exitCode -eq 0
                Output = ""
                ExitCode = $exitCode
            }
        }
    }
    catch {
        return @{
            Success = $false
            Output = $_.Exception.Message
            ExitCode = 1
        }
    }
}

# Get Git information
function Get-GitInfo {
    $info = @{}
    
    # Get remote repository URL
    $result = Invoke-SafeCommand "git remote get-url origin"
    if ($result.Success -and $result.Output) {
        $url = $result.Output.Trim()
        if ($url -match "github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$") {
            $info.Username = $matches[1]
            $info.Repository = $matches[2]
        }
    }
    
    # Get current branch
    $result = Invoke-SafeCommand "git branch --show-current"
    if ($result.Success -and $result.Output) {
        $info.Branch = $result.Output.Trim()
    } else {
        $info.Branch = "main"
    }
    
    # Get user information
    $result = Invoke-SafeCommand "git config user.name"
    if ($result.Success -and $result.Output) {
        $info.AuthorName = $result.Output.Trim()
    }
    
    $result = Invoke-SafeCommand "git config user.email"
    if ($result.Success -and $result.Output) {
        $info.AuthorEmail = $result.Output.Trim()
    }
    
    return $info
}

# Check environment
function Test-Environment {
    Write-Info "Checking environment dependencies..."
    
    $checks = @{}
    
    # Check Git
    $checks.Git = Test-Command "git"
    if ($checks.Git) {
        $result = Invoke-SafeCommand "git --version"
        Write-Success "Git: $($result.Output)"
    } else {
        Write-Error "Git: Not installed"
    }
    
    # Check Python
    $checks.Python = Test-Command "python"
    if ($checks.Python) {
        $result = Invoke-SafeCommand "python --version"
        Write-Success "Python: $($result.Output)"
    } else {
        Write-Error "Python: Not installed"
    }
    
    # Check Ruby
    $checks.Ruby = Test-Command "ruby"
    if ($checks.Ruby) {
        $result = Invoke-SafeCommand "ruby --version"
        Write-Success "Ruby: $($result.Output)"
    } else {
        Write-Warning "Ruby: Not installed (optional for local development)"
    }
    
    # Check Bundler
    $checks.Bundler = Test-Command "bundle"
    if ($checks.Bundler) {
        $result = Invoke-SafeCommand "bundle --version"
        Write-Success "Bundler: $($result.Output)"
    } else {
        Write-Warning "Bundler: Not installed"
    }
    
    return $checks
}

# Check project structure
function Test-ProjectStructure {
    Write-Info "Checking project structure..."
    
    $docsDir = Join-Path $PWD "."
    if (-not (Test-Path $docsDir)) {
        Write-Error "docs directory not found"
        return $false
    }
    
    $requiredFiles = @("_config.yml", "index.md", "Gemfile")
    $allExist = $true
    
    foreach ($file in $requiredFiles) {
        $filePath = Join-Path $docsDir $file
        if (Test-Path $filePath) {
            Write-Success "Found: $file"
        } else {
            Write-Error "Missing: $file"
            $allExist = $false
        }
    }
    
    return $allExist
}

# Get configuration interactively
function Get-InteractiveConfig {
    Write-Info "Interactive configuration..."
    
    $gitInfo = Get-GitInfo
    $config = @{}
    
    # GitHub username
    $defaultUsername = if ($Username) { $Username } elseif ($gitInfo.Username) { $gitInfo.Username } else { "" }
    if ($defaultUsername) {
        $input = Read-Host "GitHub username [$defaultUsername]"
        $config.Username = if ($input) { $input } else { $defaultUsername }
    } else {
        $config.Username = Read-Host "GitHub username"
    }
    
    # Repository name
    $defaultRepo = if ($Repository) { $Repository } elseif ($gitInfo.Repository) { $gitInfo.Repository } else { "VIVTransformer" }
    $input = Read-Host "Repository name [$defaultRepo]"
    $config.Repository = if ($input) { $input } else { $defaultRepo }
    
    # Project title
    $defaultTitle = if ($Title) { $Title } else { "VIVTransformer Documentation" }
    $input = Read-Host "Project title [$defaultTitle]"
    $config.Title = if ($input) { $input } else { $defaultTitle }
    
    # Project description
    $defaultDesc = if ($Description) { $Description } else { "Advanced Transformer Architecture with Vision Integration" }
    $input = Read-Host "Project description [$defaultDesc]"
    $config.Description = if ($input) { $input } else { $defaultDesc }
    
    # Author name
    $defaultAuthor = if ($AuthorName) { $AuthorName } elseif ($gitInfo.AuthorName) { $gitInfo.AuthorName } else { "" }
    if ($defaultAuthor) {
        $input = Read-Host "Author name [$defaultAuthor]"
        $config.AuthorName = if ($input) { $input } else { $defaultAuthor }
    } else {
        $config.AuthorName = Read-Host "Author name"
    }
    
    # Author email
    $defaultEmail = if ($AuthorEmail) { $AuthorEmail } elseif ($gitInfo.AuthorEmail) { $gitInfo.AuthorEmail } else { "" }
    if ($defaultEmail) {
        $input = Read-Host "Author email [$defaultEmail]"
        $config.AuthorEmail = if ($input) { $input } else { $defaultEmail }
    } else {
        $config.AuthorEmail = Read-Host "Author email"
    }
    
    # Branch name
    $defaultBranch = if ($Branch -ne "main") { $Branch } elseif ($gitInfo.Branch) { $gitInfo.Branch } else { "main" }
    $input = Read-Host "Deploy branch [$defaultBranch]"
    $config.Branch = if ($input) { $input } else { $defaultBranch }
    
    return $config
}

# Get automatic configuration
function Get-AutoConfig {
    Write-Info "Auto-detecting configuration..."
    
    $gitInfo = Get-GitInfo
    $config = @{}
    
    $config.Username = if ($Username) { $Username } elseif ($gitInfo.Username) { $gitInfo.Username } else { "vivtransformer" }
    $config.Repository = if ($Repository) { $Repository } elseif ($gitInfo.Repository) { $gitInfo.Repository } else { "VIVTransformer" }
    $config.Title = if ($Title) { $Title } else { "VIVTransformer Documentation" }
    $config.Description = if ($Description) { $Description } else { "Advanced Transformer Architecture with Vision Integration" }
    $config.AuthorName = if ($AuthorName) { $AuthorName } elseif ($gitInfo.AuthorName) { $gitInfo.AuthorName } else { "VIVTransformer Team" }
    $config.AuthorEmail = if ($AuthorEmail) { $AuthorEmail } elseif ($gitInfo.AuthorEmail) { $gitInfo.AuthorEmail } else { "team@vivtransformer.com" }
    $config.Branch = if ($Branch -ne "main") { $Branch } elseif ($gitInfo.Branch) { $gitInfo.Branch } else { "main" }
    
    Write-Success "Detected configuration:"
    Write-Info "  Username: $($config.Username)"
    Write-Info "  Repository: $($config.Repository)"
    Write-Info "  Title: $($config.Title)"
    Write-Info "  Author: $($config.AuthorName)"
    Write-Info "  Branch: $($config.Branch)"
    
    return $config
}

# Update configuration file
function Update-ConfigFile {
    param([hashtable]$Config)
    
    Write-Info "Updating configuration file..."
    
    $configFile = Join-Path $PWD "_config.yml"
    if (-not (Test-Path $configFile)) {
        Write-Error "Configuration file not found: $configFile"
        return $false
    }
    
    try {
        $content = Get-Content $configFile -Raw -Encoding UTF8
        
        # Update configuration items
        $content = $content -replace 'title: .*', "title: $($Config.Title)"
        $content = $content -replace 'description: .*', "description: $($Config.Description)"
        $content = $content -replace 'baseurl: .*', "baseurl: `"/$($Config.Repository)`""
        $content = $content -replace 'url: .*', "url: `"https://$($Config.Username).github.io`""
        $content = $content -replace '  name: .*', "  name: $($Config.AuthorName)"
        $content = $content -replace '  email: .*', "  email: $($Config.AuthorEmail)"
        $content = $content -replace '- https://github.com/yourusername/VIVTransformer', "- https://github.com/$($Config.Username)/$($Config.Repository)"
        
        Set-Content $configFile -Value $content -Encoding UTF8
        Write-Success "Configuration file updated successfully"
        return $true
    }
    catch {
        Write-Error "Failed to update configuration file: $($_.Exception.Message)"
        return $false
    }
}

# Install dependencies
function Install-Dependencies {
    Write-Info "Installing dependencies..."
    
    $docsDir = Join-Path $PWD "."
    $gemfile = Join-Path $docsDir "Gemfile"
    
    if (-not (Test-Path $gemfile)) {
        Write-Error "Gemfile not found"
        return $false
    }
    
    try {
        Push-Location $docsDir
        
        # Check and install Bundler
        if (-not (Test-Command "bundle")) {
            Write-Info "Installing Bundler..."
            $result = Invoke-SafeCommand "gem install bundler" -CaptureOutput $false
            if (-not $result.Success) {
                Write-Error "Failed to install Bundler"
                return $false
            }
        }
        
        # Install dependencies
        Write-Info "Installing Jekyll dependencies..."
        $result = Invoke-SafeCommand "bundle install" -CaptureOutput $false
        if ($result.Success) {
            Write-Success "Dependencies installed successfully"
            return $true
        } else {
            Write-Error "Failed to install dependencies"
            return $false
        }
    }
    finally {
        Pop-Location
    }
}

# Test local build
function Test-LocalBuild {
    Write-Info "Testing local build..."
    
    $docsDir = Join-Path $PWD "."
    
    try {
        Push-Location $docsDir
        
        $result = Invoke-SafeCommand "bundle exec jekyll build --dry-run"
        if ($result.Success) {
            Write-Success "Local build test passed"
            return $true
        } else {
            Write-Error "Local build test failed: $($result.Output)"
            return $false
        }
    }
    finally {
        Pop-Location
    }
}

# Create deployment instructions
function New-DeploymentInstructions {
    param([hashtable]$Config)
    
    Write-Info "Creating deployment instructions..."
    
    $instructions = "# GitHub Pages Deployment Instructions`n`n"
    $instructions += "## Configuration Complete!`n`n"
    $instructions += "### Configuration Info`n"
    $instructions += "- GitHub Username: $($Config.Username)`n"
    $instructions += "- Repository: $($Config.Repository)`n"
    $instructions += "- Deploy Branch: $($Config.Branch)`n"
    $instructions += "- Website URL: https://$($Config.Username).github.io/$($Config.Repository)`n`n"
    $instructions += "### Next Steps`n`n"
    $instructions += "#### 1. Commit code to GitHub`n"
    $instructions += "``````bash`n"
    $instructions += "git add .`n"
    $instructions += "git commit -m `"Configure GitHub Pages documentation site`"`n"
    $instructions += "git push origin $($Config.Branch)`n"
    $instructions += "``````n`n"
    $instructions += "#### 2. Enable GitHub Pages`n"
    $instructions += "1. Visit: https://github.com/$($Config.Username)/$($Config.Repository)/settings/pages`n"
    $instructions += "2. Select 'Deploy from a branch' in Source section`n"
    $instructions += "3. Choose '$($Config.Branch)' branch and '/docs' folder`n"
    $instructions += "4. Click 'Save'`n`n"
    $instructions += "#### 3. Wait for deployment`n"
    $instructions += "- Deployment usually takes 2-10 minutes`n"
    $instructions += "- Check deployment status in Actions tab`n"
    $instructions += "- Visit: https://$($Config.Username).github.io/$($Config.Repository)`n`n"
    $instructions += "### Local Development`n"
    $instructions += "``````bash`n"
    $instructions += "cd docs`n"
    $instructions += "bundle exec jekyll serve`n"
    $instructions += "# Visit http://localhost:4000`n"
    $instructions += "``````n`n"
    
    $instructionsFile = Join-Path $PWD "GITHUB_PAGES_SETUP.md"
    try {
        Set-Content $instructionsFile -Value $instructions -Encoding UTF8
        Write-Success "Deployment instructions saved to: $instructionsFile"
        return $true
    }
    catch {
        Write-Error "Failed to save deployment instructions: $($_.Exception.Message)"
        return $false
    }
}

# Show current status
function Show-Status {
    Write-Info "Current Configuration Status"
    Write-Host "=" * 50
    
    # Check configuration file
    $configFile = Join-Path $PWD "_config.yml"
    if (Test-Path $configFile) {
        Write-Success "Jekyll configuration file exists"
        
        $content = Get-Content $configFile -Raw
        if ($content -match 'title: (.*)') {
            Write-Info "  Title: $($matches[1])"
        }
        if ($content -match 'baseurl: "(.*)"') {
            Write-Info "  Base URL: $($matches[1])"
        }
        if ($content -match 'url: "(.*)"') {
            Write-Info "  Website URL: $($matches[1])"
        }
    } else {
        Write-Error "Jekyll configuration file not found"
    }
    
    # Check deployment instructions
    $setupFile = Join-Path $PWD "GITHUB_PAGES_SETUP.md"
    if (Test-Path $setupFile) {
        Write-Success "Deployment instructions file exists"
    } else {
        Write-Warning "Deployment instructions file not found"
    }
    
    # Check Git status
    $result = Invoke-SafeCommand "git status --porcelain"
    if ($result.Success) {
        if ($result.Output) {
            Write-Warning "Uncommitted changes detected"
            Write-Info "  Run 'git status' for details"
        } else {
            Write-Success "Working directory is clean"
        }
    }
}

# Start local preview
function Start-LocalPreview {
    Write-Info "Starting local preview server..."
    
    $docsDir = Join-Path $PWD "."
    $gemfile = Join-Path $docsDir "Gemfile"
    
    if (-not (Test-Path $gemfile)) {
        Write-Error "Gemfile not found, please run setup first"
        return
    }
    
    if (-not (Test-Command "bundle")) {
        Write-Error "Bundler not installed, please install dependencies first"
        return
    }
    
    try {
        Push-Location $docsDir
        
        Write-Info "Starting Jekyll server..."
        Write-Info "Server will be available at: http://localhost:4000"
        Write-Info "Press Ctrl+C to stop the server"
        Write-Host ""
        
        $result = Invoke-SafeCommand "bundle exec jekyll serve --host 0.0.0.0 --port 4000 --livereload" -CaptureOutput $false
        
        if (-not $result.Success) {
            Write-Error "Failed to start server"
        }
    }
    finally {
        Pop-Location
    }
}

# Main setup process
function Start-Setup {
    param(
        [string]$SetupMode,
        [hashtable]$Config = $null
    )
    
    Write-Info "VIVTransformer GitHub Pages Setup Tool"
    Write-Host "=" * 50
    
    # Check environment
    $envChecks = Test-Environment
    
    if (-not $envChecks.Git) {
        Write-Error "Git is not installed, please install Git first"
        return $false
    }
    
    # Check project structure
    if (-not (Test-ProjectStructure)) {
        Write-Error "Project structure is incomplete, please run basic setup first"
        return $false
    }
    
    # Get configuration
    Write-Info "Configuring project information..."
    if ($Config) {
        $setupConfig = $Config
        Write-Success "Using provided configuration data"
    } elseif ($SetupMode -eq "auto") {
        $setupConfig = Get-AutoConfig
    } else {
        $setupConfig = Get-InteractiveConfig
    }
    
    # Update configuration file
    if (-not (Update-ConfigFile $setupConfig)) {
        return $false
    }
    
    # Install dependencies
    if ($envChecks.Ruby) {
        if (-not (Install-Dependencies)) {
            Write-Warning "Dependency installation failed, but continuing..."
        }
    } else {
        Write-Warning "Ruby not installed, skipping dependency installation"
    }
    
    # Test build
    if ($envChecks.Ruby -and $envChecks.Bundler) {
        Test-LocalBuild | Out-Null
    } else {
        Write-Warning "Skipping build test (missing Ruby environment)"
    }
    
    # Create deployment instructions
    New-DeploymentInstructions $setupConfig | Out-Null
    
    # Complete
    Write-Success "Configuration complete!"
    Write-Host "=" * 50
    
    Write-Info "Next steps:"
    Write-Host "1. Commit code: git add . && git commit -m 'Configure GitHub Pages' && git push" -ForegroundColor Cyan
    Write-Host "2. Enable GitHub Pages: https://github.com/$($setupConfig.Username)/$($setupConfig.Repository)/settings/pages" -ForegroundColor Cyan
    Write-Host "3. Visit website: https://$($setupConfig.Username).github.io/$($setupConfig.Repository)" -ForegroundColor Cyan
    
    Write-Info "For detailed instructions, see: GITHUB_PAGES_SETUP.md"
    
    return $true
}

# Show menu
function Show-Menu {
    while ($true) {
        Clear-Host
        Write-Host "VIVTransformer GitHub Pages Setup Tool" -ForegroundColor Magenta
        Write-Host "=" * 50 -ForegroundColor Magenta
        Write-Host ""
        Write-Host "Please select setup mode:" -ForegroundColor White
        Write-Host ""
        Write-Host "[1] Quick Setup (auto-detect Git info)" -ForegroundColor Green
        Write-Host "[2] Interactive Setup (manual input)" -ForegroundColor Blue
        Write-Host "[3] View Current Configuration" -ForegroundColor Cyan
        Write-Host "[4] Install Dependencies" -ForegroundColor Yellow
        Write-Host "[5] Test Local Preview" -ForegroundColor Magenta
        Write-Host "[0] Exit" -ForegroundColor Red
        Write-Host ""
        
        $choice = Read-Host "Please enter option [0-5]"
        
        switch ($choice) {
            "1" {
                Start-Setup "auto"
                Read-Host "`nPress any key to continue..."
            }
            "2" {
                Start-Setup "interactive"
                Read-Host "`nPress any key to continue..."
            }
            "3" {
                Show-Status
                Read-Host "`nPress any key to continue..."
            }
            "4" {
                $envChecks = Test-Environment
                if ($envChecks.Ruby) {
                    Install-Dependencies
                } else {
                    Write-Error "Ruby not installed, cannot install Jekyll dependencies"
                    Write-Info "Please visit https://rubyinstaller.org/ to download and install Ruby"
                }
                Read-Host "`nPress any key to continue..."
            }
            "5" {
                Start-LocalPreview
                Read-Host "`nPress any key to continue..."
            }
            "0" {
                Write-Host "`nGoodbye!" -ForegroundColor Green
                return
            }
            default {
                Write-Warning "Invalid option, please try again"
                Start-Sleep 1
            }
        }
    }
}

# Main program entry
function Main {
    try {
        switch ($Mode) {
            "auto" {
                $config = $null
                if ($Username -or $Repository -or $Title -or $Description -or $AuthorName -or $AuthorEmail) {
                    $config = @{
                        Username = $Username
                        Repository = $Repository
                        Title = $Title
                        Description = $Description
                        AuthorName = $AuthorName
                        AuthorEmail = $AuthorEmail
                        Branch = $Branch
                    }
                }
                Start-Setup "auto" $config
            }
            "interactive" {
                $config = $null
                if ($Username -or $Repository -or $Title -or $Description -or $AuthorName -or $AuthorEmail) {
                    $config = @{
                        Username = $Username
                        Repository = $Repository
                        Title = $Title
                        Description = $Description
                        AuthorName = $AuthorName
                        AuthorEmail = $AuthorEmail
                        Branch = $Branch
                    }
                }
                Start-Setup "interactive" $config
            }
            "install" {
                $envChecks = Test-Environment
                if ($envChecks.Ruby) {
                    Install-Dependencies
                } else {
                    Write-Error "Ruby not installed, cannot install Jekyll dependencies"
                }
            }
            "preview" {
                Start-LocalPreview
            }
            "status" {
                Show-Status
            }
            "menu" {
                Show-Menu
            }
        }
    }
    catch {
        Write-Error "Error during setup: $($_.Exception.Message)"
        exit 1
    }
}

# Run main program
Main