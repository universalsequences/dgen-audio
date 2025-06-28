import XCTest
@testable import DGen

final class BlocksTests: XCTestCase {
    
    // MARK: - Topological Sort Tests
    
    func testTopoSimpleLinearGraph() throws {
        // Create a simple linear graph: 0 -> 1 -> 2
        let g = Graph()
        let node0 = g.n(.constant(1.0))  // 0
        let node1 = g.n(.add, node0, g.n(.constant(2.0)))  // 1
        let node2 = g.n(.output(0), node1)  // 2
        
        let sorted = topo(g)
        
        // Topo returns in standard topological order (dependencies come before dependents)
        XCTAssertEqual(sorted.count, 4) // includes the constant(2.0) node
        
        // Dependencies should come before dependents: constants before add, add before output
        let node2Index = sorted.firstIndex(of: node2)!
        let node1Index = sorted.firstIndex(of: node1)!
        let node0Index = sorted.firstIndex(of: node0)!
        
        XCTAssertLessThan(node0Index, node1Index) // constant before add
        XCTAssertLessThan(node1Index, node2Index) // add before output
    }
    
    func testTopoParallelPaths() throws {
        // Create a graph with parallel paths that converge:
        //     0 -> 2
        //        \/
        //        3
        //        /\
        //     1 -> 2
        let g = Graph()
        let node0 = g.n(.constant(1.0))  // 0
        let node1 = g.n(.constant(2.0))  // 1
        let node2 = g.n(.add, node0, node1)  // 2 (depends on both 0 and 1)
        let node3 = g.n(.output(0), node2)  // 3
        
        let sorted = topo(g)
        
        XCTAssertEqual(sorted.count, 4)
        
        // Dependencies should come before dependents
        let node3Index = sorted.firstIndex(of: node3)!
        let node2Index = sorted.firstIndex(of: node2)!
        let node0Index = sorted.firstIndex(of: node0)!
        let node1Index = sorted.firstIndex(of: node1)!
        
        // Constants (node0, node1) should come before add (node2)
        XCTAssertLessThan(node0Index, node2Index)
        XCTAssertLessThan(node1Index, node2Index)
        // Add (node2) should come before output (node3)
        XCTAssertLessThan(node2Index, node3Index)
    }
    
    func testTopoDisconnectedComponents() throws {
        // Create a graph with two disconnected components
        let g = Graph()
        
        // Component 1: 0 -> 1 -> 2
        let comp1_0 = g.n(.constant(1.0))
        let comp1_1 = g.n(.add, comp1_0, g.n(.constant(2.0)))
        let comp1_2 = g.n(.output(0), comp1_1)
        
        // Component 2: 3 -> 4 (disconnected)
        let comp2_0 = g.n(.constant(3.0))
        let comp2_1 = g.n(.phasor(0), comp2_0, g.n(.constant(0.0)))
        
        let sorted = topo(g)
        
        XCTAssertEqual(sorted.count, 7) // 7 nodes total
        
        // Both components should be present
        XCTAssertTrue(sorted.contains(comp1_0))
        XCTAssertTrue(sorted.contains(comp1_1))
        XCTAssertTrue(sorted.contains(comp1_2))
        XCTAssertTrue(sorted.contains(comp2_0))
        XCTAssertTrue(sorted.contains(comp2_1))
    }
    
    func testTopoWithDebugOutput() throws {
        let g = Graph()
        let node0 = g.n(.constant(1.0))
        let node1 = g.n(.output(0), node0)
        
        // Test with debug enabled
        let sorted = topo(g, debug: true)
        
        XCTAssertEqual(sorted.count, 2)
        XCTAssertTrue(sorted.contains(node0))
        XCTAssertTrue(sorted.contains(node1))
    }
    
    // MARK: - Block Determination Tests
    
    func testDetermineBlocksSimpleSIMD() throws {
        // Create a simple SIMD graph
        let g = Graph()
        let const1 = g.n(.constant(1.0))
        let const2 = g.n(.constant(2.0))
        let add = g.n(.add, const1, const2)
        let output = g.n(.output(0), add)
        
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        
        // Should have 1 or 2 blocks (constants might be in separate block from output)
        XCTAssertGreaterThanOrEqual(blocks.count, 1)
        XCTAssertLessThanOrEqual(blocks.count, 2)
        
        // All nodes should be in some block
        let allNodesInBlocks = Set(blocks.flatMap { $0.nodes })
        XCTAssertEqual(allNodesInBlocks.count, 4)
        XCTAssertTrue(allNodesInBlocks.contains(const1))
        XCTAssertTrue(allNodesInBlocks.contains(const2))
        XCTAssertTrue(allNodesInBlocks.contains(add))
        XCTAssertTrue(allNodesInBlocks.contains(output))
    }
    
    func testDetermineBlocksWithScalarNodes() throws {
        // Create a graph with scalar nodes (phasor)
        let g = Graph()
        let freq = g.n(.constant(440.0))
        let reset = g.n(.constant(0.0))
        let phasor = g.n(.phasor(0), freq, reset)
        let output = g.n(.output(0), phasor)
        
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        
        // Should have at least 2 blocks (SIMD for constants, scalar for phasor)
        XCTAssertGreaterThanOrEqual(blocks.count, 2)
        
        // Phasor should be in a scalar block
        var foundPhasorInScalarBlock = false
        for block in blocks {
            if block.nodes.contains(phasor) {
                XCTAssertEqual(block.kind, .scalar)
                foundPhasorInScalarBlock = true
                break
            }
        }
        XCTAssertTrue(foundPhasorInScalarBlock, "Phasor should be in a scalar block")
        
        // Output should be in the same block as phasor (our simplified rule)
        var foundOutputWithPhasor = false
        for block in blocks {
            if block.nodes.contains(phasor) && block.nodes.contains(output) {
                foundOutputWithPhasor = true
                break
            }
        }
        XCTAssertTrue(foundOutputWithPhasor, "Output should be in same block as its dependency (phasor)")
    }
    
    func testDetermineBlocksUnconnectedNodes() throws {
        // Test the scenario that was causing issues: unconnected nodes
        let g = Graph()
        
        // Connected component
        let freq1 = g.n(.constant(1.0))
        let reset1 = g.n(.constant(0.0))
        let phasor1 = g.n(.phasor(0), freq1, reset1)
        let output = g.n(.output(0), phasor1)
        
        // Unconnected component
        let freq2 = g.n(.constant(120.0))
        let reset2 = g.n(.constant(0.0))
        let phasor2 = g.n(.phasor(1), freq2, reset2)
        
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g, debug: true)
        
        // Should have multiple blocks
        XCTAssertGreaterThanOrEqual(blocks.count, 2)
        
        // All nodes should be assigned to blocks
        let allNodesInBlocks = Set(blocks.flatMap { $0.nodes })
        XCTAssertEqual(allNodesInBlocks.count, 7) // 7 total nodes
        
        // Both phasors should be in scalar blocks
        var phasor1BlockKind: Kind?
        var phasor2BlockKind: Kind?
        
        for block in blocks {
            if block.nodes.contains(phasor1) {
                phasor1BlockKind = block.kind
            }
            if block.nodes.contains(phasor2) {
                phasor2BlockKind = block.kind
            }
        }
        
        XCTAssertEqual(phasor1BlockKind, .scalar)
        XCTAssertEqual(phasor2BlockKind, .scalar)
    }
    
    func testDetermineBlocksWithNodeLimit() throws {
        // Test maxNodesPerBlock limiting
        let g = Graph()
        
        // Create many constants
        var constants: [NodeID] = []
        for i in 0..<10 {
            constants.append(g.n(.constant(Float(i))))
        }
        
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g, maxNodesPerBlock: 3)
        
        // Should have multiple blocks due to node limit
        XCTAssertGreaterThanOrEqual(blocks.count, 4) // 10 nodes / 3 per block = at least 4 blocks
        
        // Each block should not exceed the limit
        for block in blocks {
            XCTAssertLessThanOrEqual(block.nodes.count, 3)
        }
        
        // All constants should be assigned
        let allNodesInBlocks = Set(blocks.flatMap { $0.nodes })
        for constant in constants {
            XCTAssertTrue(allNodesInBlocks.contains(constant))
        }
    }
    
    func testDetermineBlocksDependencyOrderPreserved() throws {
        // Test that block dependencies are properly maintained
        let g = Graph()
        let const1 = g.n(.constant(1.0))
        let const2 = g.n(.constant(2.0))
        let freq = g.n(.mul, const1, const2)
        let reset = g.n(.constant(0.0))
        let phasor = g.n(.phasor(0), freq, reset)
        let output = g.n(.output(0), phasor)
        
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        
        // Find which blocks contain which nodes
        var constBlock: Int?
        var phasorBlock: Int?
        
        for (blockIdx, block) in blocks.enumerated() {
            if block.nodes.contains(const1) || block.nodes.contains(const2) {
                constBlock = blockIdx
            }
            if block.nodes.contains(phasor) {
                phasorBlock = blockIdx
            }
        }
        
        XCTAssertNotNil(constBlock)
        XCTAssertNotNil(phasorBlock)
        
        // Test dependency sorting
        let sortedBlockIds = sortBlocksByDependencies(blocks, g)
        XCTAssertFalse(sortedBlockIds.isEmpty, "Block sorting should not fail")
        
        // Constants block should come before phasor block in execution order
        let constBlockPos = sortedBlockIds.firstIndex(of: constBlock!)
        let phasorBlockPos = sortedBlockIds.firstIndex(of: phasorBlock!)
        
        XCTAssertNotNil(constBlockPos)
        XCTAssertNotNil(phasorBlockPos)
        XCTAssertLessThan(constBlockPos!, phasorBlockPos!)
    }
    
    func testDetermineBlocksWithDebugOutput() throws {
        let g = Graph()
        let node0 = g.n(.constant(1.0))
        let node1 = g.n(.output(0), node0)
        
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        
        // Test with debug enabled
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g, debug: true)
        
        XCTAssertGreaterThanOrEqual(blocks.count, 1)
        
        // All nodes should be assigned
        let allNodesInBlocks = Set(blocks.flatMap { $0.nodes })
        XCTAssertTrue(allNodesInBlocks.contains(node0))
        XCTAssertTrue(allNodesInBlocks.contains(node1))
    }
    
    // MARK: - Block Dependency Sorting Tests
    
    func testSortBlocksByDependenciesSimple() throws {
        let g = Graph()
        let const1 = g.n(.constant(1.0))
        let const2 = g.n(.constant(2.0))
        let add = g.n(.add, const1, const2)
        let phasor = g.n(.phasor(0), add, g.n(.constant(0.0)))
        let output = g.n(.output(0), phasor)
        
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        
        let sortedBlockIds = sortBlocksByDependencies(blocks, g)
        
        XCTAssertFalse(sortedBlockIds.isEmpty)
        XCTAssertEqual(sortedBlockIds.count, blocks.count)
        
        // Should contain all block indices
        for i in 0..<blocks.count {
            XCTAssertTrue(sortedBlockIds.contains(i))
        }
    }
    
    func testSortBlocksByDependenciesWithDebug() throws {
        let g = Graph()
        let node0 = g.n(.constant(1.0))
        let node1 = g.n(.phasor(0), node0, g.n(.constant(0.0)))
        let node2 = g.n(.output(0), node1)
        
        let sorted = topo(g)
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        
        // Test with debug enabled
        let sortedBlockIds = sortBlocksByDependencies(blocks, g, debug: true)
        
        XCTAssertFalse(sortedBlockIds.isEmpty)
        XCTAssertEqual(sortedBlockIds.count, blocks.count)
    }
    
    // MARK: - Edge Cases and Error Conditions
    
    func testEmptyGraph() throws {
        let g = Graph()
        
        let sorted = topo(g)
        XCTAssertTrue(sorted.isEmpty)
        
        let scalar = scalarNodes(g)
        XCTAssertTrue(scalar.isEmpty)
        
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        XCTAssertTrue(blocks.isEmpty)
    }
    
    func testSingleNodeGraph() throws {
        let g = Graph()
        let node = g.n(.constant(1.0))
        
        let sorted = topo(g)
        XCTAssertEqual(sorted.count, 1)
        XCTAssertEqual(sorted[0], node)
        
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks[0].nodes.count, 1)
        XCTAssertEqual(blocks[0].nodes[0], node)
        XCTAssertEqual(blocks[0].kind, .simd)
    }
    
    func testScalarNodeIdentification() throws {
        let g = Graph()
        
        // Create various node types
        let constant = g.n(.constant(1.0))
        let add = g.n(.add, constant, g.n(.constant(2.0)))
        let phasor = g.n(.phasor(0), add, g.n(.constant(0.0)))
        let latch = g.n(.latch(1), phasor, g.n(.constant(1.0)))
        let output = g.n(.output(0), latch)
        
        let scalar = scalarNodes(g)
        
        // Phasor and latch should be scalar
        XCTAssertTrue(scalar.contains(phasor))
        XCTAssertTrue(scalar.contains(latch))
        
        // Constants and add should not be scalar
        XCTAssertFalse(scalar.contains(constant))
        XCTAssertFalse(scalar.contains(add))
        XCTAssertFalse(scalar.contains(output))
    }
    
    // MARK: - Integration Tests
    
    func testCompleteWorkflow() throws {
        // Test the complete workflow: topo -> scalar -> blocks -> sort
        let g = Graph()
        
        // Build a complex graph similar to main.swift
        let freq1 = g.n(.mul, g.n(.constant(90)), g.n(.constant(2.5)))
        let freq2 = g.n(.mul, g.n(.constant(2)), g.n(.constant(3.5)))
        let ph1 = g.n(.phasor(0), freq1, g.n(.constant(0)))
        let ph2 = g.n(.phasor(1), freq2, g.n(.constant(0)))
        let cond = g.n(.lt, ph2, g.n(.constant(0.1)))
        let latched = g.n(.latch(2), ph1, cond)
        let mult = g.n(.mul, latched, ph1)
        let output = g.n(.output(0), mult)
        
        // Step 1: Topological sort
        let sorted = topo(g)
        XCTAssertEqual(sorted.count, g.nodes.count)
        
        // Step 2: Identify scalar nodes
        let scalar = scalarNodes(g)
        XCTAssertTrue(scalar.contains(ph1))
        XCTAssertTrue(scalar.contains(ph2))
        XCTAssertTrue(scalar.contains(latched))
        
        // Step 3: Determine blocks
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        XCTAssertGreaterThanOrEqual(blocks.count, 2)
        
        // Step 4: Sort blocks by dependencies
        let sortedBlockIds = sortBlocksByDependencies(blocks, g)
        XCTAssertEqual(sortedBlockIds.count, blocks.count)
        XCTAssertFalse(sortedBlockIds.isEmpty)
        
        // Verify all nodes are assigned to blocks
        let allNodesInBlocks = Set(blocks.flatMap { $0.nodes })
        XCTAssertEqual(allNodesInBlocks.count, g.nodes.count)
        
        for nodeId in g.nodes.keys {
            XCTAssertTrue(allNodesInBlocks.contains(nodeId), "Node \(nodeId) should be in some block")
        }
    }
    
    func testPerformanceWithLargeGraph() throws {
        // Performance test with a larger graph
        let g = Graph()
        
        // Create a chain of 100 nodes
        var lastNode = g.n(.constant(1.0))
        for _ in 0..<99 {
            lastNode = g.n(.add, lastNode, g.n(.constant(1.0)))
        }
        let output = g.n(.output(0), lastNode)
        
        measure {
            let sorted = topo(g)
            let scalar = scalarNodes(g)
            let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
            let _ = sortBlocksByDependencies(blocks, g)
        }
        
        // Ensure correctness even with large graph
        let sorted = topo(g)
        XCTAssertEqual(sorted.count, g.nodes.count)
        
        let scalar = scalarNodes(g)
        let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g)
        let sortedBlockIds = sortBlocksByDependencies(blocks, g)
        
        XCTAssertFalse(sortedBlockIds.isEmpty)
        
        let allNodesInBlocks = Set(blocks.flatMap { $0.nodes })
        XCTAssertEqual(allNodesInBlocks.count, g.nodes.count)
    }
}