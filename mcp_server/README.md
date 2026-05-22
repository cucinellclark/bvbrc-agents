# BV-BRC MCP Server

A Model Context Protocol (MCP) server for the **Bacterial-Viral Bioinformatics Resource Center (BV-BRC)**:

- **Data Tools**: Query BV-BRC Solr collections for genome, feature, and other biological data
- **Service Tools**: Submit and manage BV-BRC analysis jobs (assembly, annotation, BLAST, etc.)
- **Workspace Tools**: Manage BV-BRC workspace files, folders, and groups

> **Note:** This repository has historically also carried README content for the **BV-BRC Copilot API** (a separate Node/Express service). That documentation is preserved below under **“BV-BRC Copilot API (reference)”**.

---

<details>
<summary><h2>Features</h2></summary>

### Data Tools
- `query_collection`: Query any BV-BRC Solr collection with flexible filtering
- `solr_collection_parameters`: Get schema information for collections
- `solr_query_instructions`: Get help on query syntax
- `solr_collections`: List all available collections

### Service Tools
- `list_service_apps`: List all available BV-BRC analysis services
- `get_job_details`: Query the status of submitted jobs
- Submit jobs for various analyses:
  - Genome Assembly
  - Genome Annotation
  - Comprehensive Genome Analysis
  - BLAST
  - Primer Design
  - Variation Analysis
  - TnSeq
  - Phylogenetic Trees (Bacterial Genome Tree, Gene Tree)
  - SNP Analysis (Whole Genome, MSA)
  - Metagenomics (Taxonomic Classification, Binning, Read Mapping)
  - RNA-Seq
  - Viral Services (SARS-CoV-2 Analysis, Sequence Submission)
  - And many more...

### Workspace Tools
- `workspace_browse_tool`: Unified browse/search tool (search recursively or inspect path as folder listing/file metadata)
- `get_file_metadata`: Get normalized metadata for workspace files or local session files
- `workspace_download_file_tool`: Download workspace files
- `workspace_upload`: Upload files to workspace
- `create_genome_group`: Create genome groups
- `create_feature_group`: Create feature groups
- `get_genome_group_ids`: Get genome IDs from a group
- `get_feature_group_ids`: Get feature IDs from a group

</details>

<details>
<summary><h2>Connecting the Remote BV-BRC MCP Server to ChatGPT</h2></summary>

### Step 1: Enable Developer Mode
1. Click the **plus** next to "Ask me anything"
2. Click **"Add sources"**
3. You should now see "Sources" and "Add" below your chat box
4. Click the **down arrow** next to "Add"
5. Click **"Connect more"**
6. Scroll down to **Advanced Settings**
7. Click the toggle next to **Developer Mode** (must be "on")
8. Click **Back**

### Step 2: Create MCP Server Connection
1. In the upper right-hand corner, click **"Create"**
2. Fill in the following:
   - **Icon**: Optional
   - **Name**: BV-BRC MCP
   - **Description**: ''
   - **MCP Server URL**: https://dev-7.bv-brc.org/mcp
3. **Authentication**:
   - Leave authentication on OAuth
4. Check the box if you **Trust this application**
5. Click **"Create"**

### Step 3: Connect to Your Server
1. You should now see 'BV-BRC MCP' under **"Enabled apps & connectors"**
2. Click the 'X' in the top left to go back to the chat screen
3. In a **New Chat**, click the '+' button and hover over **More**
  - You should see **BV-BRC MCP** as an option under **Canvas**
4. Select **BV-BRC MCP**

</details>

<details>
<summary><h2>Connecting the Remote BV-BRC MCP Server to Claude</h2></summary>

1. Click **account** in bottom left and go to **settings**
2. Click **'Connectors'**
3. Click **'Add custom connector'**
4. Fill in the following:
   - **Name**: BV-BRC MCP
   - **Remote MCP server URL**: https://dev-7.bv-brc.org/mcp
5. Click **'Add'**
6. Then click **'Connect'**
7. Log into BV-BRC
8. It's now available to use in a new chat

</details>

<details>
<summary><h2>Installing as a Claude Extension</h2></summary>

Open your terminal

0. Clone Github Repository
   ```bash
   git clone https://github.com/cucinellclark/bvbrc-mcp-server
   cd bvbrc-mcp-server
