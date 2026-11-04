// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title FederatedAuditRegistry
/// @notice Permissioned audit layer for a federated ensemble. Hospitals commit the
///         SHA-256 digest of each locally trained model, peer hospitals attest the
///         model's discrimination on their own validation data, and the contract
///         admits a model to the global ensemble only if the median peer-attested
///         AUROC reaches the round threshold. No patient data or model parameters
///         are stored on-chain; only digests, sample counts and integer scores.
contract FederatedAuditRegistry {
    struct Submission {
        bytes32 modelHash;
        bytes32 metaHash;      // digest of the training manifest (feature schema, hyper-parameters)
        uint32 nTrain;
        uint64 submittedAt;
        bool exists;
        bool accepted;
        uint16 medianScore;    // AUROC x 10^4
        uint16[] scores;
    }

    struct Round {
        uint16 threshold;      // AUROC x 10^4
        bool open;
        bool finalized;
        bytes32 globalHash;
        bytes32 weightsHash;
        address[] submitters;
    }

    address public immutable governor;
    address public aggregator;
    mapping(address => bytes32) public siteId;
    mapping(address => bool) public isNode;
    uint256 public nodeCount;

    mapping(uint256 => Round) private rounds;
    mapping(uint256 => mapping(address => Submission)) private subs;
    mapping(uint256 => mapping(address => mapping(address => bool))) public attested;

    event NodeRegistered(address indexed node, bytes32 siteId);
    event RoundOpened(uint256 indexed round, uint16 threshold);
    event ModelSubmitted(uint256 indexed round, address indexed node, bytes32 modelHash, uint32 nTrain);
    event Attested(uint256 indexed round, address indexed validator, address indexed owner, uint16 score, bytes32 evalHash);
    event ModelDecision(uint256 indexed round, address indexed node, bool accepted, uint16 medianScore);
    event GlobalCommitted(uint256 indexed round, bytes32 globalHash, bytes32 weightsHash);

    modifier onlyGovernor() { require(msg.sender == governor, "not governor"); _; }
    modifier onlyAggregator() { require(msg.sender == aggregator, "not aggregator"); _; }
    modifier onlyNode() { require(isNode[msg.sender], "unregistered node"); _; }

    constructor(address _aggregator) {
        governor = msg.sender;
        aggregator = _aggregator;
    }

    function registerNode(address node, bytes32 id) external onlyGovernor {
        require(!isNode[node], "already registered");
        isNode[node] = true;
        siteId[node] = id;
        nodeCount += 1;
        emit NodeRegistered(node, id);
    }

    function openRound(uint256 r, uint16 threshold) external onlyAggregator {
        require(!rounds[r].open && !rounds[r].finalized, "round exists");
        rounds[r].threshold = threshold;
        rounds[r].open = true;
        emit RoundOpened(r, threshold);
    }

    function submitModel(uint256 r, bytes32 modelHash, bytes32 metaHash, uint32 nTrain) external onlyNode {
        require(rounds[r].open, "round closed");
        Submission storage s = subs[r][msg.sender];
        require(!s.exists, "duplicate submission");
        s.modelHash = modelHash;
        s.metaHash = metaHash;
        s.nTrain = nTrain;
        s.submittedAt = uint64(block.timestamp);
        s.exists = true;
        rounds[r].submitters.push(msg.sender);
        emit ModelSubmitted(r, msg.sender, modelHash, nTrain);
    }

    /// @notice A validating hospital reports the AUROC (x 10^4) it measured for `owner`'s
    ///         model on its own validation split. The evaluated digest must match the
    ///         committed one, so a validator cannot score a different artefact.
    function attest(uint256 r, address owner, bytes32 evaluatedHash, uint16 score, bytes32 evalHash) external onlyNode {
        require(rounds[r].open, "round closed");
        require(owner != msg.sender, "self attestation");
        Submission storage s = subs[r][owner];
        require(s.exists, "no submission");
        require(s.modelHash == evaluatedHash, "hash mismatch");
        require(!attested[r][msg.sender][owner], "duplicate attestation");
        require(score <= 10000, "score range");
        attested[r][msg.sender][owner] = true;
        s.scores.push(score);
        emit Attested(r, msg.sender, owner, score, evalHash);
    }

    function finalizeRound(uint256 r) external onlyAggregator {
        Round storage rd = rounds[r];
        require(rd.open, "round closed");
        rd.open = false;
        rd.finalized = true;
        for (uint256 i = 0; i < rd.submitters.length; i++) {
            Submission storage s = subs[r][rd.submitters[i]];
            uint16 m = _median(s.scores);
            s.medianScore = m;
            s.accepted = s.scores.length > 0 && m >= rd.threshold;
            emit ModelDecision(r, rd.submitters[i], s.accepted, m);
        }
    }

    function commitGlobal(uint256 r, bytes32 globalHash, bytes32 weightsHash) external onlyAggregator {
        Round storage rd = rounds[r];
        require(rd.finalized, "not finalized");
        require(rd.globalHash == bytes32(0), "already committed");
        rd.globalHash = globalHash;
        rd.weightsHash = weightsHash;
        emit GlobalCommitted(r, globalHash, weightsHash);
    }

    // ----- read-only verification used by hospitals and auditors -----

    function verifyModel(uint256 r, address node, bytes32 h) external view returns (bool) {
        Submission storage s = subs[r][node];
        return s.exists && s.accepted && s.modelHash == h;
    }

    function getSubmission(uint256 r, address node)
        external view returns (bytes32, bytes32, uint32, bool, uint16, uint256)
    {
        Submission storage s = subs[r][node];
        return (s.modelHash, s.metaHash, s.nTrain, s.accepted, s.medianScore, s.scores.length);
    }

    function getRound(uint256 r) external view returns (uint16, bool, bool, bytes32, bytes32, uint256) {
        Round storage rd = rounds[r];
        return (rd.threshold, rd.open, rd.finalized, rd.globalHash, rd.weightsHash, rd.submitters.length);
    }

    function _median(uint16[] storage a) private view returns (uint16) {
        uint256 n = a.length;
        if (n == 0) return 0;
        uint16[] memory b = new uint16[](n);
        for (uint256 i = 0; i < n; i++) b[i] = a[i];
        for (uint256 i = 1; i < n; i++) {
            uint16 key = b[i];
            uint256 j = i;
            while (j > 0 && b[j - 1] > key) { b[j] = b[j - 1]; j--; }
            b[j] = key;
        }
        if (n % 2 == 1) return b[n / 2];
        return uint16((uint256(b[n / 2 - 1]) + uint256(b[n / 2])) / 2);
    }
}
